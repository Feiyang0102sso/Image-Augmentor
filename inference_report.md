# Image Augmentation Inference Report

![Visualization](demo_picture/inference_report.png)

## Prediction Results

### Image 1
- **Original**: sturgeon (55.37%)
- **RandomHorizontalFlip(p=0.5)**: sturgeon (55.37%)
- **RandomRotation(angle_range=[100, 50])**: tench (43.39%)
- **BrightnessAdjust(delta_range=[30, 60])**: tench (46.55%)
- **ContrastAdjust(factor_range=[1.5, 2.5])**: dugong (16.65%)

