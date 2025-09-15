# train_lightning.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# 从你的文件中导入模型和数据集
from geocnn import GeoCNN
from dataset import WarpDataset, seed_worker
from utils import initialize_flow, flow_to_image, upsample_flow

import torch.multiprocessing as mp

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

class DewarpingModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, batch_size=4):
        super().__init__()
        self.save_hyperparameters() # 保存超参数, 如 lr, batch_size

        self.model = GeoCNN()
        checkpoint = torch.load("/home/lixumin/project/docDewarp/my_model/ckpt/best_light_model-loss0.0037.pth", map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_fn = nn.L1Loss() # L1 损失对光流任务通常更鲁棒

    def forward(self, image, coords):
        return self.model(image, coords)

    def training_step(self, batch, batch_idx):
        image, gt_flow, _ = batch # 数据集返回 (new_im, flow_up, flow_up)
        
        # 1. 初始化坐标网格
        _, coords0, coords1 = initialize_flow(image)
        coords1 = coords1.detach() # 初始坐标网格不需要梯度

        # 2. 模型预测
        # 模型输出的是掩码和更新后的坐标网格
        pred_mask, pred_coords1 = self.forward(image, coords1)
        
        # 3. 计算预测的光流
        # 光流 = 预测的坐标 - 原始坐标
        pred_flow = pred_coords1 - coords0
        
        pred_flow = upsample_flow(pred_flow, pred_mask)
        # 4. 计算损失
        loss = self.loss_fn(pred_flow, gt_flow)
        
        # 5. 记录日志
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, gt_flow, _ = batch
        
        _, coords0, coords1 = initialize_flow(image)
        coords1 = coords1.detach()
        
        pred_mask, pred_coords1 = self.forward(image, coords1)
        pred_flow = pred_coords1 - coords0
        pred_flow = upsample_flow(pred_flow, pred_mask)
        
        loss = self.loss_fn(pred_flow, gt_flow)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # --- 可视化 ---
        # 只在每个 epoch 的第一个 batch 上进行可视化，避免日志文件过大
        if batch_idx == 0:
            # 取 batch 中的第一个样本
            img_sample = image[0].cpu().numpy().transpose(1, 2, 0)
            pred_flow_sample = pred_flow[0].cpu().numpy()
            gt_flow_sample = gt_flow[0].cpu().numpy()

            # 将光流转换为可视化图像
            pred_flow_viz = flow_to_image(pred_flow_sample)
            gt_flow_viz = flow_to_image(gt_flow_sample)
            
            # 使用 TensorBoard logger 记录图像
            self.logger.experiment.add_image('val/input_image', img_sample, self.global_step, dataformats='HWC')
            self.logger.experiment.add_image('val/predicted_flow', pred_flow_viz, self.global_step, dataformats='HWC')
            self.logger.experiment.add_image('val/ground_truth_flow', gt_flow_viz, self.global_step, dataformats='HWC')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        
        # 使用学习率调度器 (可选，但推荐)
        # OneCycleLR 是一个很好的选择
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # 每个 step 更新学习率
            },
        }


def main():
    mp.set_start_method('spawn', force=True)
    # --- 超参数 ---
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    NUM_WORKERS = 8
    
    # 为了演示，我们使用少量样本。在实际训练中，请增大 sample 数量
    TRAIN_SAMPLES = 10000
    VAL_SAMPLES = 32

    # --- 设定随机种子 ---
    pl.seed_everything(42, workers=True)

    # --- 数据集和数据加载器 ---
    # 注意: WarpDataset 会动态生成数据，这可能会成为训练瓶颈。
    # 在实际项目中，最好是先生成一个固定的数据集再进行训练。
    full_dataset = WarpDataset(
        warp_image_paths=["/home/lixumin/project/data/Vary-600k/data"],
        dewarps_image_paths=["/home/lixumin/project/data/question-data/train"],
        bg_image_paths=["/home/lixumin/project/nanoseg/data/bg_images"],
        sample=TRAIN_SAMPLES + VAL_SAMPLES
    )

    # 划分训练集和验证集
    train_dataset, val_dataset = random_split(full_dataset, [TRAIN_SAMPLES, VAL_SAMPLES])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # --- 模型 ---
    model = DewarpingModule(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

    # --- 日志和回调 ---
    logger = TensorBoardLogger("tb_logs", name="geocnn_dewarp")
    
    # 模型检查点：保存验证损失最低的模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='geocnn-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )
    
    # 学习率监视器
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- 训练器 ---
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0, 1], # 使用第一个 GPU
        max_epochs=NUM_EPOCHS,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16, # 使用混合精度训练以加速并减少显存占用
    )

    # --- 开始训练 ---
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl_model = DewarpingModule.load_from_checkpoint(
        "/home/lixumin/project/doclayout-pipeline/dewarp/checkpoints/geocnn-epoch=09-val_loss=1.0567.ckpt",
        map_location=device
    )
    model = pl_model.model
    model.to(device)
    model.eval()
    # 3. 获取该模型的 state_dict (状态字典)
    model_weights = model.state_dict()

    # 4. 使用 torch.save() 将权重保存为 .pt 文件
    output_pt_path = "dewarping_model_weights.pt"
    torch.save(model_weights, output_pt_path)

    

    

if __name__ == "__main__":
    # !!! 修改为你自己的数据集路径 !!!
    # 示例路径，请替换
    # "/home/lixumin/project/data/Vary-600k/data/pdf_data/pdf_cn_30w/114126"
    # "/home/lixumin/project/data/question-data/3L2H6Q"
    # main()

    test()