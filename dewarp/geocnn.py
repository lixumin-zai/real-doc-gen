import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 辅助类和函数 (来自原代码，已添加注释) ---


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch'):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(192, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(192, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        return x


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = .25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1




def coords_grid(batch, ht, wd, device=None):
    """
    生成一个坐标网格。
    Args:
        batch (int): 批量大小。
        ht (int): 网格高度。
        wd (int): 网格宽度。
        device: 指定张量所在的设备。
    Returns:
        coords (Tensor): 形状为 [batch, 2, ht, wd] 的坐标网格。
                         coords[:, 0, :, :] 是 x 坐标。
                         coords[:, 1, :, :] 是 y 坐标。
    """
    # 如果未指定设备，则使用CPU
    if device is None:
        device = torch.device('cpu')
    # 生成一维的高度和宽度坐标
    coords_y, coords_x = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    # 将y和x坐标堆叠起来，并转换为浮点数
    coords = torch.stack([coords_x, coords_y], dim=0).float()
    # 在最前面增加一个维度，并复制batch次，以匹配批量大小
    return coords[None].repeat(batch, 1, 1, 1)

# --- 新的CNN模型 ---

class GeoCNN(nn.Module):
    """
    一个基于纯CNN的U-Net架构模型，用于替代原有的GeoTr。
    """
    def __init__(self):
        # 初始化模型
        super(GeoCNN, self).__init__()

        # 定义隐藏层维度
        self.hidden_dim = hdim = 256
        # 定义特征提取器，输出维度为hdim。使用'instance'归一化
        self.fnet = BasicEncoder(output_dim=hdim, norm_fn='instance')

        # --- U-Net 编码器路径 ---
        # 编码器块1 (在 1/8 分辨率下)
        self.encoder_block1 = nn.Sequential(
            ResidualBlock(hdim, hdim, norm_fn='instance'),
            ResidualBlock(hdim, hdim, norm_fn='instance')
        )
        # 下采样层1 (1/8 -> 1/16)
        self.down_layer1 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=2, padding=1)
        
        # 编码器块2 (在 1/16 分辨率下)
        self.encoder_block2 = nn.Sequential(
            ResidualBlock(hdim, hdim, norm_fn='instance'),
            ResidualBlock(hdim, hdim, norm_fn='instance')
        )
        # 下采样层2 (1/16 -> 1/32)
        self.down_layer2 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=2, padding=1)

        # 瓶颈层 (在 1/32 分辨率下)
        self.bottleneck_block = nn.Sequential(
            ResidualBlock(hdim, hdim, norm_fn='instance'),
            ResidualBlock(hdim, hdim, norm_fn='instance')
        )

        # --- U-Net 解码器路径 ---
        # 上采样层1 (1/32 -> 1/16)
        self.up_layer1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 解码器块1 (处理拼接后的特征)
        # 输入通道为 hdim(来自上采样) + hdim(来自跳跃连接) = 2*hdim
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(2 * hdim, hdim, kernel_size=3, padding=1), # 融合层
            ResidualBlock(hdim, hdim, norm_fn='instance')
        )

        # 上采样层2 (1/16 -> 1/8)
        self.up_layer2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 解码器块2 (处理拼接后的特征)
        # 输入通道为 hdim(来自上采样) + hdim(来自跳跃连接) = 2*hdim
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(2 * hdim, hdim, kernel_size=3, padding=1), # 融合层
            ResidualBlock(hdim, hdim, norm_fn='instance')
        )

        self.output_conv = nn.Conv2d(hdim, hdim, kernel_size=1)

        # 实例化更新块，接收解码器最终输出的特征图
        self.update_block = UpdateBlock(self.hidden_dim)

    def forward(self, image1, coords1, return_features=False):
        """
        模型的前向传播过程。
        """
        # 1. 初始特征提取
        fmap = self.fnet(image1) # -> [B, hdim, H/8, W/8]
        fmap = torch.relu(fmap)  # 应用ReLU激活
        # print(fmap)
        # 2. U-Net 编码器路径
        skip1 = self.encoder_block1(fmap) # 经过第一个编码块，得到跳跃连接1 -> [B, hdim, H/8, W/8]
        down1 = self.down_layer1(skip1)   # 下采样 -> [B, hdim, H/16, W/16]
        
        skip2 = self.encoder_block2(down1) # 经过第二个编码块，得到跳跃连接2 -> [B, hdim, H/16, W/16]
        down2 = self.down_layer2(skip2)    # 下采样 -> [B, hdim, H/32, W/32]
        
        bottleneck = self.bottleneck_block(down2) # 经过瓶颈块 -> [B, hdim, H/32, W/32]

        # 3. U-Net 解码器路径
        up1 = self.up_layer1(bottleneck) # 上采样 -> [B, hdim, H/16, W/16]
        # 拼接上采样特征和跳跃连接2
        concat1 = torch.cat([up1, skip2], dim=1) # -> [B, 2*hdim, H/16, W/16]
        dec1 = self.decoder_block1(concat1) # 经过第一个解码块 -> [B, hdim, H/16, W/16]

        up2 = self.up_layer2(dec1) # 上采样 -> [B, hdim, H/8, W/8]
        # 拼接上采样特征和跳跃连接1
        concat2 = torch.cat([up2, skip1], dim=1) # -> [B, 2*hdim, H/8, W/8]
        # 得到最终的特征图输出
        fmap_decoded = self.decoder_block2(concat2) # -> [B, hdim, H/8, W/8]
        fmap_out = self.output_conv(fmap_decoded)
        # 4. 光流预测和上采样
        # 使用更新块预测掩码和更新后的低分辨率光流坐标
        mask, coords1 = self.update_block(fmap_out, coords1)
        # 根据标志决定是否返回中间特征图
        if return_features:
            return mask, coords1, fmap_out
        # 默认只返回最终的密集对应场
        return mask, coords1