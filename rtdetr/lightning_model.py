import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
from model import RTDETR
from utils import HungarianMatcher, SetCriterion
import torch

class LitRTDETR(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-4, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()
        
        # 我们的模型
        self.model = RTDETR(num_classes=num_classes, num_queries=300, backbone='resnet18')

        # 匹配器
        matcher = HungarianMatcher()

        # 损失权重
        self.weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 2}
        
        # 损失函数
        # eos_coef: 对"no object"类别在分类损失中的权重
        self.criterion = SetCriterion(num_classes, matcher, self.weight_dict, eos_coef=0.1)

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        
        # 计算损失
        loss_dict = self.criterion(outputs, targets)
        
        # 计算加权总损失
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
        
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        # 日志记录
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        # 日志记录
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f'val_{k}', v.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.hparams.lr, 
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6  # 学习率衰减到的最小值
        )
        
        # 返回 Lightning 需要的字典格式
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 'epoch' 表示每个 epoch 结束时更新 lr
                'frequency': 1,
            }
        }
