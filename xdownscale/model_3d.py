import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        r = self.upscale_factor
        b, c, d, h, w = x.size()
        out_c = c // (r ** 3)

        if c % (r ** 3) != 0:
            raise ValueError(f"Channel dimension {c} is not divisible by upscale_factor^3 = {r**3}")

        x = x.view(b, out_c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, out_c, d * r, h * r, w * r)
        return x

class SRCNN3D(nn.Module):
    def __init__(self):
        super(SRCNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=4)   # Feature extraction
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)  # Non-linear mapping
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding=2)   # Reconstruction

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # No activation on final layer
        return x

class FSRCNN3D(nn.Module):
    def __init__(self, d=56, s=12, m=4, upscale_factor=1):
        super(FSRCNN3D, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv3d(1, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )

        self.mid_parts = [nn.Sequential(
            nn.Conv3d(d, s, kernel_size=1),
            nn.PReLU(s)
        )]

        for _ in range(m - 1):
            self.mid_parts.append(nn.Sequential(
                nn.Conv3d(s, s, kernel_size=3, padding=3//2),
                nn.PReLU(s)
            ))

        self.mid_parts = nn.Sequential(*self.mid_parts)

        self.last_part = nn.Sequential(
            nn.Conv3d(s, d, kernel_size=1),
            nn.PReLU(d),
            nn.Conv3d(d, 1 * upscale_factor**3, kernel_size=3, padding=1)
        )

        self.pixel_shuffle = PixelShuffle3D(upscale_factor)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_parts(x)
        x = self.last_part(x)
        x = self.pixel_shuffle(x)
        return x

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CARNM3D(nn.Module):
    def __init__(self, num_channels=1, scale_factor=1, num_residual_groups=2, num_residual_blocks=2, num_channels_rg=64):
        super(CARNM3D, self).__init__()
        self.scale_factor = scale_factor

        self.entry = ConvBlock3D(num_channels, num_channels_rg, kernel_size=3, stride=1, padding=1)

        self.residual_groups = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock3D(num_channels_rg, num_channels_rg, kernel_size=3, stride=1, padding=1)
                for _ in range(num_residual_blocks)
            ])
            for _ in range(num_residual_groups)
        ])

        if scale_factor > 1:
            self.upsample = nn.ConvTranspose3d(num_channels_rg, num_channels_rg, kernel_size=3,
                                               stride=scale_factor, padding=1,
                                               output_padding=scale_factor - 1)
        else:
            self.upsample = None

        self.exit = ConvBlock3D(num_channels_rg, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.entry(x)
        residuals = [rg(x) for rg in self.residual_groups]
        x = sum(residuals)
        if self.upsample is not None:
            x = self.upsample(x)
        return self.exit(x)


class LapSRN3D(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(LapSRN3D, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv3d(64, in_channels * upscale_factor**3, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = PixelShuffle3D(upscale_factor) if upscale_factor > 1 else nn.Identity()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

class FALSRB3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32, scale_factor=1):
        super(FALSRB3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.residual = self.make_layer(num_features, num_features, 3)

        self.conv2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        if scale_factor > 1:
            self.upsample = nn.ConvTranspose3d(
                num_features, out_channels, kernel_size=3,
                stride=scale_factor, padding=1, output_padding=scale_factor - 1
            )
        else:
            self.upsample = nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)

    def make_layer(self, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.residual(x1)
        x3 = self.relu2(self.conv2(x1 + x2))
        out = self.upsample(x3)
        return out

class ResidualBlock3D(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual

class CARN3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, upscale_factor=1):
        super(CARN3D, self).__init__()
        self.entry = nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1)

        self.b1 = ResidualBlock3D(num_features)
        self.b2 = ResidualBlock3D(num_features)
        self.b3 = ResidualBlock3D(num_features)

        self.c1 = nn.Conv3d(num_features * 2, num_features, kernel_size=1)
        self.c2 = nn.Conv3d(num_features * 3, num_features, kernel_size=1)
        self.c3 = nn.Conv3d(num_features * 4, num_features, kernel_size=1)

        if upscale_factor > 1:
            self.upsample = nn.ConvTranspose3d(
                num_features, num_features, kernel_size=3, stride=upscale_factor,
                padding=1, output_padding=upscale_factor - 1
            )
        else:
            self.upsample = None

        self.exit = nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.entry(x)

        x2 = self.b1(x1)
        x2 = self.c1(torch.cat([x1, x2], dim=1))

        x3 = self.b2(x2)
        x3 = self.c2(torch.cat([x1, x2, x3], dim=1))

        x4 = self.b3(x3)
        x4 = self.c3(torch.cat([x1, x2, x3, x4], dim=1))

        if self.upsample is not None:
            x4 = self.upsample(x4)

        out = self.exit(x4)
        return out

class FALSR_A3D(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(FALSR_A3D, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(32, in_channels * (upscale_factor ** 3), kernel_size=3, stride=1, padding=1)

        self.pixel_shuffle = PixelShuffle3D(upscale_factor)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.pixel_shuffle(self.conv6(x5))
        return x6

class OISRRK2_3D(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(OISRRK2_3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, in_channels * (upscale_factor ** 3), kernel_size=3, stride=1, padding=1)
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = PixelShuffle3D(upscale_factor)

    def forward(self, x):
        res1 = F.relu(self.conv1(x))
        res2 = F.relu(self.conv2(res1))
        res3 = F.relu(self.conv3(res2))
        res4 = F.relu(self.conv4(res3))
        res5 = self.conv5(res4)
        out = self.pixel_shuffle(res5) + x
        return out

class ResidualBlock3D_BN(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock3D_BN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

class MDSR3D(nn.Module):
    def __init__(self, in_channels, upscale_factor, num_blocks):
        super(MDSR3D, self).__init__()
        self.input_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.residual_blocks = nn.Sequential(*[ResidualBlock3D_BN(64) for _ in range(num_blocks)])
        self.output_conv = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.prelu(x1)
        x2 = self.residual_blocks(x1)
        x3 = self.output_conv(x2)
        return x + x3

class SecondOrderChannelAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SecondOrderChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels)
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * torch.sigmoid(y)

class SAN3D(nn.Module):
    def __init__(self, in_channels, upscale_factor, num_blocks, num_heads):
        super(SAN3D, self).__init__()

        self.input_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock3D_BN(64) for _ in range(num_blocks)]
        )

        self.attention_blocks = nn.Sequential(
            *[SecondOrderChannelAttention3D(64) for _ in range(num_heads)]
        )

        self.output_conv = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.prelu(x1)

        x2 = self.residual_blocks(x1)
        x3 = self.attention_blocks(x2)
        x4 = self.output_conv(x3)

        return x + x4


class ResidualChannelAttentionBlock3D(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResidualChannelAttentionBlock3D, self).__init__()
        modules_body = []
        for _ in range(2):
            modules_body.append(nn.Conv3d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
            if bn: modules_body.append(nn.BatchNorm3d(n_feat))
            modules_body.append(act)
        modules_body.pop()  # remove last activation
        self.body = nn.Sequential(*modules_body)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(n_feat, n_feat // reduction, 1, padding=0, bias=bias),
            act,
            nn.Conv3d(n_feat // reduction, n_feat, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res) * res
        res += x
        return res

class RCAN3D(nn.Module):
    def __init__(self, in_channels, num_blocks, upscale_factor):
        super(RCAN3D, self).__init__()
        self.input_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.residual_blocks = nn.Sequential(
            *[ResidualChannelAttentionBlock3D(64) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv3d(64, in_channels * (upscale_factor ** 3), kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle3D(upscale_factor)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.prelu(x1)
        x2 = self.residual_blocks(x1)
        x3 = self.output_conv(x2)
        x4 = self.pixel_shuffle(x3)
        return x4
