import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List
import math

# --- 1. 辅助模块 ---

class ConvBlock(nn.Module):
    """一个简单的卷积块，用于CCFM中的特征融合"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
    """构建二维正弦余弦位置编码"""
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    # Corrected meshgrid indexing for PyTorch
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    
    dim_t = torch.arange(embed_dim // 4, dtype=torch.float32)
    inv_freq = 1.0 / (temperature ** (dim_t / (embed_dim // 4)))
    
    pos_w = grid_w.unsqueeze(-1) * inv_freq
    pos_h = grid_h.unsqueeze(-1) * inv_freq
    
    pos_w = torch.cat([pos_w.sin(), pos_w.cos()], dim=-1)
    pos_h = torch.cat([pos_h.sin(), pos_h.cos()], dim=-1)
    
    pos_embed = torch.cat([pos_h, pos_w], dim=-1).permute(2, 0, 1).unsqueeze(0)
    return pos_embed

# --- 2. 主干网络 ---

class Backbone(nn.Module):
    """
    使用torchvision的create_feature_extractor来提取中间特征
    对应原文中的C3, C4, C5
    """
    def __init__(self, model_name='resnet50'):
        super().__init__()
        
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            # 对于ResNet50, C3, C4, C5 分别是 layer2, layer3, layer4 的输出
            # ResNet50 使用 Bottleneck block, 最后一个激活层名称为 relu_2
            self.return_nodes = {
                'layer2.3.relu_2': 'c3',
                'layer3.5.relu_2': 'c4',
                'layer4.2.relu_2': 'c5',
            }
            # 输出通道数
            self.out_channels = [512, 1024, 2048]
            
        elif model_name == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # 对于ResNet18, C3, C4, C5 分别是 layer2, layer3, layer4 的输出
            # ResNet18 使用 BasicBlock, 最后一个激活层名称为 relu
            self.return_nodes = {
                'layer2.1.relu': 'c3',
                'layer3.1.relu': 'c4',
                'layer4.1.relu': 'c5',
            }
            # 输出通道数
            self.out_channels = [128, 256, 512]
            
        elif model_name == 'mobilenet_v2':
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            # MobileNetV2的特征提取器在 features 属性中
            # 我们需要找到 stride 为 8, 16, 32 的层
            self.return_nodes = {
                'features.6.conv.0': 'c3',  # stride = 8
                'features.13.conv.0': 'c4', # stride = 16
                'features.17.conv.0': 'c5', # stride = 32
            }
            # 对应层的输出通道数
            self.out_channels = [32, 96, 320]

        elif model_name == 'mobilenet_v3_large':
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            # MobileNetV3_Large 的特征提取器也在 features 属性中
            self.return_nodes = {
                'features.4': 'c3',  # stride = 8
                'features.7': 'c4',  # stride = 16
                'features.13': 'c5', # stride = 32
            }
            # 对应层的输出通道数
            self.out_channels = [40, 80, 160]
            
        else:
            raise NotImplementedError(f"Backbone {model_name} not supported")
        
        self.body = create_feature_extractor(model, self.return_nodes)

    def forward(self, x):
        features = self.body(x)
        
        return list(features.values())

# --- 3. 颈部网络 (Hybrid Encoder) ---


class HybridEncoder(nn.Module):
    """
    RT-DETR的核心：混合编码器 (JIT-Trace Friendly Version)
    包含 AIFI 和 CCFM
    通过手动展开循环来确保与 torch.jit.trace 的兼容性。
    """
    def __init__(self, in_channels=[512, 1024, 2048], hidden_dim=256, nhead=8, dim_feedforward=1024):
        super().__init__()
        
        # 1. 将 ModuleList 替换为具体命名的层
        # 输入投射层 (p stands for projection)
        self.input_proj_c3 = nn.Sequential(
            nn.Conv2d(in_channels[0], hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        self.input_proj_c4 = nn.Sequential(
            nn.Conv2d(in_channels[1], hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        self.input_proj_c5 = nn.Sequential(
            nn.Conv2d(in_channels[2], hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        # AIFI Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True 
        )
        self.aifi = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        
        # CCFM: PaFPN (手动定义每一层)
        # FPN (Top-down) layers
        self.lateral_conv_p4 = ConvBlock(hidden_dim, hidden_dim, kernel_size=1) # for p5 -> p4
        self.fpn_block_p4 = ConvBlock(hidden_dim * 2, hidden_dim)
        
        self.lateral_conv_p3 = ConvBlock(hidden_dim, hidden_dim, kernel_size=1) # for p4 -> p3
        self.fpn_block_p3 = ConvBlock(hidden_dim * 2, hidden_dim)
            
        # PAN (Bottom-up) layers
        self.downsample_conv_n3 = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=2) # for n3 -> n4
        self.pan_block_n4 = ConvBlock(hidden_dim * 2, hidden_dim)
        
        self.downsample_conv_n4 = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=2) # for n4 -> n5
        self.pan_block_n5 = ConvBlock(hidden_dim * 2, hidden_dim)


    def forward(self, feats: List[torch.Tensor]):
        # 假设 feats 是 [c3, c4, c5]
        c3, c4, c5 = feats
        
        # 2. 手动展开输入投射
        p3 = self.input_proj_c3(c3)
        p4 = self.input_proj_c4(c4)
        p5 = self.input_proj_c5(c5)
        
        # AIFI Module on C5
        b, c, h, w = p5.shape

        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(p5.device)
        
        src_flatten = p5.flatten(2).permute(0, 2, 1)
        pos_embed_flatten = pos_embed.flatten(2).permute(0, 2, 1)
        
        combined_input = src_flatten+pos_embed_flatten
        memory = self.aifi(combined_input)
        
        p5_aifi = memory.permute(0, 2, 1).reshape(b, c, h, w)

        # 3. 手动展开 CCFM (PaFPN)
        # --- FPN: Top-down path ---
        # P5 -> P4
        p5_lat = self.lateral_conv_p4(p5_aifi)
        p5_upsampled = F.interpolate(p5_lat, size=p4.shape[2:], mode='nearest')
        fused_p4 = torch.cat([p5_upsampled, p4], dim=1)
        p4_fpn = self.fpn_block_p4(fused_p4)

        # P4 -> P3
        p4_lat = self.lateral_conv_p3(p4_fpn)
        p4_upsampled = F.interpolate(p4_lat, size=p3.shape[2:], mode='nearest')
        fused_p3 = torch.cat([p4_upsampled, p3], dim=1)
        p3_fpn = self.fpn_block_p3(fused_p3)
        # FPN outputs are: p3_fpn, p4_fpn, p5_aifi
        
        # --- PAN: Bottom-up path ---
        # N3 (p3_fpn) -> N4
        n3_downsampled = self.downsample_conv_n3(p3_fpn)
        fused_n4 = torch.cat([n3_downsampled, p4_fpn], dim=1)
        p4_pan = self.pan_block_n4(fused_n4)

        # N4 (p4_pan) -> N5
        n4_downsampled = self.downsample_conv_n4(p4_pan)
        fused_n5 = torch.cat([n4_downsampled, p5_aifi], dim=1)
        p5_pan = self.pan_block_n5(fused_n5)

        # 最终输出与原逻辑一致，分别是PAN在P3, P4, P5尺度上的结果
        # N3 is p3_fpn, N4 is p4_pan, N5 is p5_pan
        # p3_fpn = torch.randn(1, 128, 80, 80)
        # p4_pan = torch.randn(1, 128, 40, 40)
        # p5_pan = torch.randn(1, 128, 20, 20)
        # print(p3_fpn.shape, p4_pan.shape, p5_pan.shape)
        return [p3_fpn, p4_pan, p5_pan]

# --- 4. 解码器和预测头 ---

class RTDETRDecoder(nn.Module):
    def __init__(self, num_queries=300, hidden_dim=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.num_queries = num_queries
        self.object_queries = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, multi_scale_feats: List[torch.Tensor]):
        memory_list = []
        pos_embed_list = []
        for feat in multi_scale_feats:
            b, c, h, w = feat.shape
            pos = build_2d_sincos_position_embedding(w, h, c).to(feat.device)
            memory_list.append(feat.flatten(2).permute(0, 2, 1))
            pos_embed_list.append(pos.flatten(2).permute(0, 2, 1))
        
        memory = torch.cat(memory_list, dim=1)
        pos_embed = torch.cat(pos_embed_list, dim=1)
        
        query_embed = self.object_queries.weight.unsqueeze(0).repeat(memory.size(0), 1, 1)
        tgt = torch.zeros_like(query_embed)

        # --- FIX START ---
        # Decoder forward: 将位置编码直接加到 memory 和 tgt 上
        memory_with_pos = memory + pos_embed
        # 在DETR中，query_embed既是tgt的位置编码，也参与到初始tgt的生成中
        tgt_with_pos = tgt + query_embed
        
        decoder_output = self.decoder(tgt=tgt_with_pos, memory=memory_with_pos)
        # --- FIX END ---
        
        return decoder_output

class RTDETRHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        self.class_head = nn.Linear(hidden_dim, num_classes+1)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, decoder_output):
        class_logits = self.class_head(decoder_output)
        pred_boxes = self.bbox_head(decoder_output).sigmoid()
        return class_logits, pred_boxes
        
# --- 5. 整合完整的 RT-DETR 模型 ---

class RTDETR(nn.Module):
    def __init__(self, num_classes=10, num_queries=300, backbone='mobilenet_v3_large'):
        super().__init__()
        self.backbone = Backbone(model_name=backbone)
        self.hidden_dim = 128
        self.hybrid_encoder = HybridEncoder(
            in_channels=self.backbone.out_channels, 
            hidden_dim=self.hidden_dim
        )
        self.decoder = RTDETRDecoder(
            num_queries=num_queries,
            hidden_dim=self.hidden_dim,
            num_layers=6
        )
        self.rtdetr_head = RTDETRHead(hidden_dim=self.hidden_dim, num_classes=num_classes)

    def forward(self, x):
        backbone_feats = self.backbone(x)

        encoder_feats = self.hybrid_encoder(backbone_feats)
        # return encoder_feats
        decoder_output = self.decoder(encoder_feats)
        # return decoder_output
        class_logits, pred_boxes = self.rtdetr_head(decoder_output)
        # return class_logits, pred_boxes
        return {'pred_logits': class_logits, 'pred_boxes': pred_boxes}


class model:
    def __init__(self):
        self.detr = RTDETR(num_classes=10, num_queries=300, backbone='resnet18')
        # self.detr.eval()

    def save_model(self):
        torch.save(self.detr.state_dict(), "./model.pt")

    def save_trace(self):
        dummy_input = torch.randn(1, 3, 640, 640)
        traced_model = torch.jit.trace(self.detr, dummy_input)
        traced_model.save("./output/trace.pt")

    def load_model(self):
        self.detr.load_state_dict(torch.load("./model.pt"))


    def get_model_parameters(self):
        num_params = sum(p.numel() for p in self.detr.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params / 1e6:.2f} M")

    def __call__(self, dummy_input):
        outputs = self.detr(dummy_input)
        return outputs

# --- 6. 使用示例 ---
if __name__ == '__main__':

    detr = model()

    # detr.load_model()
    # detr.save_model() 
    detr.save_trace()  

    torch.manual_seed(0)
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = detr(dummy_input)   
        print(outputs) 

    print(detr.get_model_parameters())

    
    # cd output && pnnx ./trace.pt inputshape=[1,3,640,640] && cd ..