# Model Architecture Ablation Study

## Comparison on ISIC 2018 Skin Lesion Dataset
| Architecture | Backing Engine | Params (M) | FLOPs (G) | Dice (%) | IoU (%) |
|---|---|---|---|---|---|
| UNet | Conv2D | 34.5 | 65.2 | 87.2 | 78.4 |
| Swin-UNet | Swin Transformer | 27.1 | 51.4 | 89.1 | 80.9 |
| TransUNet | Hybrid ViT | 105.3 | 98.7 | 89.7 | 81.5 |
| **VM-UNetV2 (Ours)** | **SS2D Visual Mamba** | **22.4** | **31.8** | **91.4** | **84.2** |
