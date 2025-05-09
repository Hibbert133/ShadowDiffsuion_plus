import os
import numpy as np
from PIL import Image
import glob
import cv2

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
# from skimage.color import rgb2lab

def compute_mse(img1, img2):
    """
    计算 MSE (Mean Squared Error)
    """
    return mse(img1.flatten(), img2.flatten())

def compute_rmse(img1, img2):
    """
    计算 RMSE (Root Mean Squared Error)
    """
    return np.sqrt(mse(img1.flatten(), img2.flatten()))


def main():
    # -----------------------------
    # 1. 修改以下三个目录为你的实际路径
    # # -----------------------------
    # maskdir = './istdplus/results/'
    # # maskdir = './data/ISTD_Dataset/test/test_B/'
     
    # shadowdir = './istdplus/results/'
    # freedir = './istdplus/results/'

    
    
    maskdir = './srd3/results/'
    # shadowdir = './srd3/results/'
    # freedir = './srd3/results/'

    #maskdir = './results/SRD/results/'

     
    shadowdir = './results/SRD/results/'
    freedir = './results/SRD/results/'

     #maskdir = './results/SRD/results/'

     
    # shadowdir = './results/ISTD+/results/'
    # freedir = './results/ISTD+/results/'


    # 获取文件路径
    mask_files = sorted(glob.glob(os.path.join(maskdir, '*_inf.png')))
    shadow_files = sorted(glob.glob(os.path.join(shadowdir, '*_sr.png')))
    free_files = sorted(glob.glob(os.path.join(freedir, '*_hr.png')))

    assert len(mask_files) == len(shadow_files) == len(free_files), \
        "文件数量不匹配，请检查文件名是否一一对应。"

    result_txt_path = 'sd_istdplus.txt'
    with open(result_txt_path, 'w') as f:
        f.write("Filename\tPSNR\tSSIM\tRMSE\n")

    # 统计量
    psnr_val_rgb, psnr_val_ns, psnr_val_s = [], [], []
    ssim_val_rgb, ssim_val_ns, ssim_val_s = [], [], []
    rmse_val_rgb, rmse_val_ns, rmse_val_s = [], [], []
    mse_val_rgb, mse_val_ns, mse_val_s = [], [], []

    for i in range(len(shadow_files)):
        sname = shadow_files[i]
        fname = free_files[i]
        mname = mask_files[i]

        # 读取图像
        rgb_restored = np.array(Image.open(sname).convert('RGB'), dtype=np.float32) / 255.0
        rgb_gt = np.array(Image.open(fname).convert('RGB'), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mname), dtype=np.float32) / 255.0

        # 调整大小到 (256,256)
        # rgb_restored = cv2.resize(rgb_restored, (256, 256))
        # rgb_gt = cv2.resize(rgb_gt, (256, 256))
        # mask = cv2.resize(mask, (256, 256))

        if len(mask.shape) == 3:
            # mask = mask[:, :, 0]  # 转换为单通道
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('mask.png', (mask * 255).astype(np.uint8))  # 保存为灰度图像


        # Binarize mask
        bm = np.where(mask == 0, 0, 1).astype(np.float32)  # 二值化掩码
        bm = np.expand_dims(bm, axis=2)
        # print(bm.shape)
        #bm=mask

        # 保存图像
        cv2.imwrite('rgb_restored.png', (rgb_restored * 255).astype(np.uint8))  # 保存彩色图像
        cv2.imwrite('rgb_gt.png', (rgb_gt * 255).astype(np.uint8))  # 保存彩色图像

        # 保存 mask（单通道处理）
        # print(mask.shape)
  

        # print(mask.shape)

        

        # Grayscale 转换
        gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
        #print(gray_restored.shape)
        # print(gray_restored.shape)
        gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
        # print(gray_gt.shape)

        # # 确保 bm 的形状与灰度图像匹配
        # if bm.shape != gray_restored.shape:
        #     bm = cv2.resize(bm, (gray_restored.shape[1], gray_restored.shape[0]))

        # 计算 SSIM
        # mask_ns = 1 - bm.squeeze()
        # print("gray_restored:", gray_restored.shape)
        # print("gray_gt:", gray_gt.shape)
        # print("mask_ns:", mask_ns.shape)
        ssim_val_rgb.append(ssim(gray_restored, gray_gt, channel_axis=None))
        ssim_val_ns.append(ssim(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None))
        ssim_val_s.append(ssim(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None))

        # 计算 PSNR
        psnr_val_rgb.append(psnr(rgb_restored, rgb_gt))
        psnr_val_ns.append(psnr(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
        psnr_val_s.append(psnr(rgb_restored * bm, rgb_gt * bm))

        # # 计算 MSE
        # mse_val_rgb.append(compute_mse(rgb_restored, rgb_gt))
        # mse_val_ns.append(compute_mse(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
        # mse_val_s.append(compute_mse(rgb_restored * bm, rgb_gt * bm))

        # 计算 RMSE (LAB 空间)


        rmse_temp = np.abs(cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3
        rmse_val_rgb.append(rmse_temp)
        rmse_temp_s = np.abs(cv2.cvtColor(rgb_restored * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum()
        rmse_temp_ns = np.abs(cv2.cvtColor(rgb_restored * (1-bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * (1-bm),
                                                                                                   cv2.COLOR_RGB2LAB)).sum() / (1-bm).sum()


        rmse_val_s.append(rmse_temp_s)
        rmse_val_ns.append(rmse_temp_ns)

                # 每张图像指标输出
        # print(f"Image: {os.path.basename(sname)}")
        # print(f"  PSNR   - All: {psnr_val_rgb[-1]:.4f}, Shadow: {psnr_val_s[-1]:.4f}, Non-Shadow: {psnr_val_ns[-1]:.4f}")
        # print(f"  SSIM   - All: {ssim_val_rgb[-1]:.4f}, Shadow: {ssim_val_s[-1]:.4f}, Non-Shadow: {ssim_val_ns[-1]:.4f}")
        # print(f"  RMSE   - All: {rmse_val_rgb[-1]:.4f}, Shadow: {rmse_val_s[-1]:.4f}, Non-Shadow: {rmse_val_ns[-1]:.4f}")

        with open(result_txt_path, 'a') as f:
            f.write(f"{os.path.basename(sname)}  PSNR : {psnr_val_rgb[-1]:.4f}  SSIM : {ssim_val_rgb[-1]:.4f}  RMSE : {rmse_val_rgb[-1]:.4f}\n")


        print(f"Processed {i+1}/{len(shadow_files)}: {os.path.basename(sname)}")
        # if i == 0:
        #     break

    # -----------------------------
    # 4. 结果输出
    # -----------------------------
        # 计算 PSNR、SSIM 和 RMSE 的均值
    
    print(len(psnr_val_rgb))
    print(len(shadow_files))
    psnr_val_rgb = sum(psnr_val_rgb) / len(shadow_files)
    ssim_val_rgb = sum(ssim_val_rgb) / len(shadow_files)
    rmse_val_rgb = sum(rmse_val_rgb) / len(shadow_files)

    psnr_val_s = sum(psnr_val_s) / len(shadow_files)
    ssim_val_s = sum(ssim_val_s) / len(shadow_files)
    rmse_val_s = sum(rmse_val_s) / len(shadow_files)

    psnr_val_ns = sum(psnr_val_ns) / len(shadow_files)
    ssim_val_ns = sum(ssim_val_ns) / len(shadow_files)
    rmse_val_ns = sum(rmse_val_ns) / len(shadow_files)

    # 打印结果
    print("PSNR: %.4f, SSIM: %.4f, RMSE: %.4f" % (psnr_val_rgb, ssim_val_rgb, rmse_val_rgb))
    print("SPSNR: %.4f, SSSIM: %.4f, SRMSE: %.4f" % (psnr_val_s, ssim_val_s, rmse_val_s))
    print("NSPSNR: %.4f, NSSSIM: %.4f, NSRMSE: %.4f" % (psnr_val_ns, ssim_val_ns, rmse_val_ns))

    # print("PSNR (all, non-shadow, shadow):")
    # print(f"{np.mean(psnr_val_rgb):.4f}\t{np.mean(psnr_val_ns):.4f}\t{np.mean(psnr_val_s):.4f}\n")

    # print("SSIM (all, non-shadow, shadow):")
    # print(f"{np.mean(ssim_val_rgb):.4f}\t{np.mean(ssim_val_ns):.4f}\t{np.mean(ssim_val_s):.4f}\n")

    # # print("MSE (all, non-shadow, shadow):")
    # # print(f"{np.mean(mse_val_rgb):.4f}\t{np.mean(mse_val_ns):.4f}\t{np.mean(mse_val_s):.4f}\n")

    # print("RMSE (all, non-shadow, shadow):")
    # print(f"{np.mean(rmse_val_rgb):.4f}\t{np.mean(rmse_val_ns):.4f}\t{np.mean(rmse_val_s):.4f}\n")

if __name__ == '__main__':
    main()
