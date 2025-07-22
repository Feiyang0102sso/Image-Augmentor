# Image Augmentation Inference Report

![Visualization](../demo_picture/single_augmented_comparison.png)

## Prediction Results

### Image 1
- **Original**: sturgeon (55.37%)
- **RandomHorizontalFlip(p=1)**: sturgeon (84.13%)
- **RandomRotation(angle_range=[100, 50])**: tench (42.88%)
- **BrightnessAdjust(delta_range=[30, 60])**: tench (55.12%)
- **ContrastAdjust(factor_range=[1.5, 2.5])**: tench (74.64%)

