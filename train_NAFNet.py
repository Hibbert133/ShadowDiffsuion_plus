import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
from model.unet_refiner import UNetRefiner
from model.nafnet import NAFNet
import core.logger as Logger
from torchvision.utils import save_image
from utils.losses import SSIMLoss, VGGPerceptualLoss
import data as Data
from model.sr3_modules import transformer,transformer2
import utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
import random

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_wandb_ckpt', action='store_true')
parser.add_argument('-log_eval', action='store_true')
args = parser.parse_args()

# -----------------------------
# Load config
# -----------------------------
opt = Logger.parse(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------

random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# Create datasets
# -----------------------------
for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train' and args.phase != 'val':
        train_set = Data.create_dataset(dataset_opt, phase)
        train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    elif phase == 'val':
        val_set = Data.create_dataset(dataset_opt, phase)
        val_loader = Data.create_dataloader(val_set, dataset_opt, phase)

# -----------------------------
# Model setup
# -----------------------------
model = NAFNet(
    img_channel=4,
    width=64,
    enc_blk_nums=[2, 2, 4, 8],
    middle_blk_num=12,
    dec_blk_nums=[2, 2, 2, 2]
).to(device)




# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt['train']['optimizer']['lr'], weight_decay=0.0, betas=(0.9, 0.999))

l1_loss = nn.L1Loss()
ssim_loss = SSIMLoss()
vgg_loss = VGGPerceptualLoss()



if opt['setting']['use_degradation_estimate']:
    # model_restoration = transformer2.ShadowFormer()
    # #print(model_restoration.win_size)
    # model_restoration.cuda()
    # utils.load_checkpoint(model_restoration, opt['setting']['degradation_model_path'])
    # model_restoration.eval()
    pass



# -----------------------------
# Training loop
# -----------------------------
num_epochs = opt['train'].get('epochs', 120)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs)
warmup_epochs=5
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
save_dir = opt['path'].get('experiments_root', './experiments')
os.makedirs(save_dir, exist_ok=True)
best_psnr=0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        input_img = data['SR'].to(device)
        mask=data['mask'].to(device)
        gt_img = data['HR'].to(device)
        x_with_mask = torch.cat([(input_img+1)/2, mask], dim=1)
        output = model(x_with_mask)
        output=torch.clamp(output, 0, 1)
        gt_img=(gt_img+1)/2     #(0,1)
        l1 = l1_loss(output, gt_img)
        ssim = ssim_loss(output, gt_img)
        vgg = vgg_loss(output, gt_img)
        mask_l1 = (torch.abs(output - gt_img) * mask.expand_as(output)).mean()
        loss = l1 + 0.1 * ssim + 0.05 * vgg + 0.2 * mask_l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        print(f"\r[Epoch {epoch+1}] Step {i+1}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}", end='', flush=True)


    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader):.4f}")
    scheduler.step()

    # Visualize
    if (epoch + 1) % 5 == 0:
        model.eval()
        psnr_val_rgb = []

        with torch.no_grad():
            for ii, data_val in enumerate(val_loader):
                val_input = data_val['SR'].to(device)           # ShadowFormer output
                gt = data_val['HR'].to(device)                  # Ground truth
                mask=data_val['mask'].to(device)
                val_input=(val_input+1)/2
                val_input = torch.cat([val_input, mask], dim=1)
                val_output = model(val_input)
                val_output = torch.clamp(val_output, 0, 1)  # residual + clamp
                gt=(gt+1)/2
                psnr_val_rgb.append(utils.batch_PSNR(val_output, gt, False).item())
            psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(save_dir, "model_best.pth"))
            
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(save_dir, "model_latest.pth"))

            print(f"[Ep {epoch+1}] PSNR: {psnr_val_rgb:.4f} | Best: {best_psnr:.4f} (Epoch {best_epoch+1})")
            sample_input = val_input[:4]
            sample_output = val_output[:4]
            sample_gt = gt[:4]
            save_image(sample_output, f"{save_dir}/epoch_{epoch+1}_refined.png")
            save_image(sample_gt, f"{save_dir}/epoch_{epoch+1}_gt.png")
