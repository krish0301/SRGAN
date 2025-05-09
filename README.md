# SRGAN Super-Resolution Pipeline (Google Colab)

> **State-of-the-art single-image super-resolution**  
> End-to-end SRGAN implementation with patch-based training and quantitative evaluation.

---

## Overview

This Colab notebook delivers a rigorously engineered SRGAN workflow conforming exactly to Ledig et al. (CVPR 2017). It comprises:

1. **Patch-Based Data Pipeline**  
   - Random 96 × 96 HR crops → 24 × 24 LR bicubic downsampling  
   - Ensures fixed-size batches for robust GPU utilization  

2. **Two-Phase Training**  
   - **Phase I (Pre-train Generator)**: 100 epochs of L2 (MSE) minimization  
   - **Phase II (Adversarial GAN)**: 200 epochs alternating generator/discriminator updates  

3. **Perceptual Loss**  
   - VGG-19 feature extractor (first 36 layers) for content loss  

4. **Automated Evaluation**  
   - Super-resolve a sample of 10 patches  
   - Compute average PSNR & SSIM, with fallback to grayscale SSIM for small patches  

---

## Repository Structure

```
/content
├── data/
│   ├── DIV2K_train_HR/        # Flattened HR images
│   └── DIV2K_train_LR/        # Generated LR images
├── checkpoints/               # Model weights (.pth)
├── results/                   # Super-resolved output samples
└── SRGAN_Colab_Notebook.ipynb # This notebook
```

---

## Setup & Execution

1. **Clone or Upload Notebook**  
   Open `SRGAN_Colab_Notebook.ipynb` in Colab.

2. **Step 1: GPU & Dependencies**  
   ```bash
   !nvidia-smi
   !pip install torch torchvision pillow scikit-image tqdm
   ```

3. **Step 2: Download & Prepare DIV2K**  
   ```bash
   # Download and flatten HR
   !wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -O DIV2K_HR.zip
   !unzip -q DIV2K_HR.zip -d temp

   # Move all images to data/DIV2K_train_HR/
   mkdir -p data/DIV2K_train_HR
   mv temp/**/*.png temp/**/*.jpg data/DIV2K_train_HR/

   # Generate LR (4× bicubic)
   python - <<EOF
   from PIL import Image
   import os
   hr_dir, lr_dir = 'data/DIV2K_train_HR','data/DIV2K_train_LR'
   os.makedirs(lr_dir, exist_ok=True)
   for fn in os.listdir(hr_dir):
       img = Image.open(f"{hr_dir}/{fn}")
       w,h = img.size
       img.resize((w//4,h//4), Image.BICUBIC).save(f"{lr_dir}/{fn}")
   EOF
   ```

4. **Step 3: Model Definitions**  
   - **Generator**: SRResNet backbone + PixelShuffle upsampling  
   - **Discriminator**: 8-layer conv → adaptive pooling → dense → sigmoid  
   - **VGG Extractor**: Pre-trained VGG-19 for perceptual loss  

5. **Step 4: Patch-Based Dataset**  
   Uses a custom `SRPatchDataset` to produce uniform 24×24 LR / 96×96 HR pairs.

6. **Step 5: Phase I – Pre-training**  
   ```python
   # 100 epochs, MSE only
   for epoch in range(100):
       for lr, hr in loader:
           sr = G(lr)
           loss = MSE(sr, hr)
           loss.backward(); optimizer_G.step()
   torch.save(G.state_dict(), 'checkpoints/srgan_pretrained.pth')
   ```

7. **Step 6: Phase II – Adversarial Training**  
   ```python
   # 200 epochs, alternating G/D
   for epoch in range(200):
       for lr, hr in loader:
           # D update
           loss_D = ½ [BCE(D(hr),1) + BCE(D(G(lr).detach()),0)]
           loss_D.backward(); optimizer_D.step()
           # G update
           loss_G = MSE(VGG(G(lr)), VGG(hr)) + 1e-3·BCE(D(G(lr)),1)
           loss_G.backward(); optimizer_G.step()
       torch.save(G.state_dict(), f'checkpoints/srgan_GAN_epoch{epoch+1}.pth')
   ```

8. **Step 7: Evaluation & Metrics**  
   - Automatically selects latest GAN checkpoint (or falls back to pre-trained).  
   - Saves 10 output patches under `results/sample_*.png`.  
   - Computes PSNR & SSIM; gracefully handles small patch SSIM via grayscale fallback.

---

## Hyperparameters

| Parameter       | Value    |
| --------------- | -------- |
| Patch size      | 96 (HR)  |
| Scale factor    | 4×       |
| Batch size      | 16       |
| Learning rate   | 1 × 10⁻⁴ |
| Pre-train epochs| 100      |
| GAN epochs      | 200      |

---

## Results

After completion, inspect `results/` for qualitative comparisons and view printed PSNR/SSIM metrics at the end of Section 8.

---

## References

1. **Ledig et al.** “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.” *CVPR*, 2017.  
2. **Wang et al.** “ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.” *ECCV*, 2018.  
3. **Wang et al.** “Real-ESRGAN: Training Real-World Degradation Models for Blind Super-Resolution.” *ICCVW*, 2021.  

---

*Prepared by [Your Name], Senior AI Engineer*
