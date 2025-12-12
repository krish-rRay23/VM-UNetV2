# Medical Image Segmentation Benchmark Protocol

## Datasets
1. **ISIC 2017 & 2018**: Skin lesion boundary segmentation.
   - Preprocessing: Bili-linear resize to 256x256, z-score normalization.
   - Evaluation metrics: Dice (DSC), Jaccard (IoU), Accuracy, Sensitivity, Specificity.
2. **Synapse Multi-Organ CT**: 30 abdominal CT scans (3779 slices).
   - Multi-organ targets: Aorta, Gallbladder, Left Kidney, Right Kidney, Liver, Pancreas, Spleen, Stomach.
   - Evaluation metrics: Average DSC and 95% Hausdorff Distance (HD95).
