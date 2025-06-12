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


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class CALayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.c1 = nn.Conv3d(channel, channel // reduction, 1, padding=0)
        self.c2 = nn.Conv3d(channel // reduction, channel, 1, padding=0)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.c1(y)
        y = nn.ReLU()(y)
        y = self.c2(y)
        return nn.Sigmoid()(y) * x

class Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block3D, self).__init__()
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CALayer3D(out_channels)

    def forward(self, x):
        h0 = self.relu(self.c1(x))
        h1 = self.c2(h0)
        h1 = self.ca(h1)
        return h1

class DLGSANet3D(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(DLGSANet3D, self).__init__()

        self.input_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            Block3D(64, 64),
            Block3D(64, 64),
            Block3D(64, 64),
            Block3D(64, 64)
        )

        self.output_conv = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.relu(self.input_conv(x))
        x2 = self.blocks(x1)
        x3 = self.output_conv(x2)
        return x + x3

class DPMN3D(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(DPMN3D, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)

        self.conv6 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(64)

        self.conv7 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm3d(64)

        self.conv8 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm3d(64)

        self.conv9 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm3d(64)

        self.conv10 = nn.Conv3d(64, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        x = self.relu(self.bn4(self.conv5(x)))
        x = self.relu(self.bn5(self.conv6(x)))
        x = self.relu(self.bn6(self.conv7(x)))
        x = self.relu(self.bn7(self.conv8(x)))
        x = self.relu(self.bn8(self.conv9(x)))

        x = self.conv10(x)
        x = torch.add(x, residual)

        return x

class SAFMN3D(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(SAFMN3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(64, in_channels * upscale_factor ** 3, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle3D(upscale_factor)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = self.conv6(x5)
        out = self.pixel_shuffle(x6)

        return out

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)
    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError(f'Unsupported padding type: {padding}. Only "same" or "valid" are supported.')

    unfold = torch.nn.Unfold(kernel_size=ksizes, padding=0, stride=strides)
    patches = unfold(images)
    return patches, paddings

class CrossAttentionSALSA3D(nn.Module):
    def __init__(self, ksize=3, stride_1=2, stride_2=2, softmax_scale=10, shape=64, p_len=64, in_channels=64,
                 inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False):
        super(CrossAttentionSALSA3D, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE

        self.conv33 = nn.Conv3d(2 * in_channels, in_channels, kernel_size=1)
        self.g = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.W = nn.Conv3d(inter_channels, in_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, inter_channels, kernel_size=1)

    def forward(self, s, g):
        output = []
        for B in range(s.shape[0]):
            b_one = s[B]  # source
            d_one = g[B]  # guide
            kernel = self.ksize

            b1 = self.g(b_one)
            b2 = self.theta(d_one)
            b3 = self.phi(d_one)
            raw_int_bs = list(b1.size())

            mid_depth = raw_int_bs[2] // 2
            unfold = torch.nn.Unfold(kernel_size=(kernel, kernel), padding=kernel // 2, stride=self.stride_1)

            patch_28 = unfold(b1[:, :, mid_depth])
            patch_28 = patch_28.view(raw_int_bs[0], self.inter_channels, kernel, kernel, -1).permute(0, 4, 1, 2, 3)

            patch_112 = unfold(b2[:, :, mid_depth])
            patch_112 = patch_112.view(raw_int_bs[0], self.inter_channels, kernel, kernel, -1).permute(0, 4, 1, 2, 3)

            patch_112_2 = unfold(b3[:, :, mid_depth])
            patch_112_2 = patch_112_2.view(raw_int_bs[0], self.inter_channels, kernel, kernel, -1).permute(0, 4, 1, 2, 3)

            q = patch_28.contiguous().view(patch_28.shape[0] * patch_28.shape[1], -1)
            k = patch_112_2.permute(2, 3, 4, 0, 1).contiguous().view(-1, patch_112_2.shape[0] * patch_112_2.shape[1])
            score_map = torch.matmul(q, k)

            b_s, l_s, d_s, h_s, w_s = b_one.shape[0], patch_28.shape[1], b_one.shape[1], b_one.shape[2], b_one.shape[3]
            att = F.softmax(score_map * self.softmax_scale, dim=1)
            v = patch_112.contiguous().view(patch_112.shape[0] * patch_112.shape[1], -1)
            attMulV = torch.mm(att, v)

            zi = attMulV.view(b_s, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[3], raw_int_bs[4]), (kernel, kernel), padding=kernel // 2, stride=self.stride_1)

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=kernel // 2, stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[3], raw_int_bs[4]), (kernel, kernel), padding=kernel // 2, stride=self.stride_1)
            zi = zi / out_mask

            zi = zi.unsqueeze(2).repeat(1, 1, raw_int_bs[2], 1, 1)  # broadcast depth dimension
            y = self.W(zi)
            y = b_one + y

            if self.add_SE:
                y_SE = self.SE(y)
                y = self.conv33(torch.cat((y_SE * y, y), dim=1))

            output.append(y)

        return torch.stack(output, dim=0)

    def GSmap(self, a, b):
        return torch.matmul(a, b)


class SALSA3D(nn.Module):
    def __init__(self, ksize=3, stride_1=2, stride_2=2, softmax_scale=10, shape=64, p_len=64, in_channels=64,
                 inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False):
        super(SALSA3D, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE

        self.conv33 = nn.Conv3d(2 * in_channels, in_channels, kernel_size=1)
        self.g = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.W = nn.Conv3d(inter_channels, in_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, inter_channels, kernel_size=1)

    def forward(self, b):
        output = []
        for B in range(b.shape[0]):
            b_one = b[B]
            kernel = self.ksize

            b1 = self.g(b_one)
            b2 = self.theta(b_one)
            b3 = self.phi(b_one)
            raw_int_bs = list(b1.size())

            mid_depth = raw_int_bs[2] // 2
            unfold = torch.nn.Unfold(kernel_size=(kernel, kernel), padding=kernel // 2, stride=self.stride_1)

            patch_28 = unfold(b1[:, :, mid_depth])
            patch_28 = patch_28.view(raw_int_bs[0], self.inter_channels, kernel, kernel, -1).permute(0, 4, 1, 2, 3)

            patch_112 = unfold(b2[:, :, mid_depth])
            patch_112 = patch_112.view(raw_int_bs[0], self.inter_channels, kernel, kernel, -1).permute(0, 4, 1, 2, 3)

            patch_112_2 = unfold(b3[:, :, mid_depth])
            patch_112_2 = patch_112_2.view(raw_int_bs[0], self.inter_channels, kernel, kernel, -1).permute(0, 4, 1, 2, 3)

            q = patch_28.contiguous().view(patch_28.shape[0] * patch_28.shape[1], -1)
            k = patch_112_2.permute(2, 3, 4, 0, 1).contiguous().view(-1, patch_112_2.shape[0] * patch_112_2.shape[1])
            score_map = torch.matmul(q, k)

            b_s, l_s, d_s, h_s, w_s = b_one.shape[0], patch_28.shape[1], b_one.shape[1], b_one.shape[2], b_one.shape[3]
            att = F.softmax(score_map * self.softmax_scale, dim=1)
            v = patch_112.contiguous().view(patch_112.shape[0] * patch_112.shape[1], -1)
            attMulV = torch.mm(att, v)

            zi = attMulV.view(b_s, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[3], raw_int_bs[4]), (kernel, kernel), padding=kernel // 2, stride=self.stride_1)

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=kernel // 2, stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[3], raw_int_bs[4]), (kernel, kernel), padding=kernel // 2, stride=self.stride_1)
            zi = zi / out_mask

            zi = zi.unsqueeze(2).repeat(1, 1, raw_int_bs[2], 1, 1)
            y = self.W(zi)
            y = b_one + y

            if self.add_SE:
                y_SE = self.SE(y)
                y = self.conv33(torch.cat((y_SE * y, y), dim=1))

            output.append(y)

        return torch.stack(output, dim=0)

    def GSmap(self, a, b):
        return torch.matmul(a, b)


class SE_net3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_net3D, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        o1 = self.pool(x)
        o1 = F.relu(self.fc1(o1))
        o1 = self.fc2(o1)
        return o1
