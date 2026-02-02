# README1

## Overview
This repository contains scripts for generating FGSM/PGD adversarial examples on images and evaluating their impact on image classification and semantic segmentation.

## Quick Start
1) Ensure required files exist:
- Input image at images/0016E5_02250.png
- ImageNet index at imagenet_class_index.json
- Segmentation checkpoint at models/deeplabv3/best_deeplabv3plus_mobilenet_cityscapes_os16.pth (only if segmentation is enabled)

2) Run the FGSM script from the project root:
- python fgsm.py

3) Optional CLI flags:
- --device auto|cpu|cuda
- --attack fgsm|pgd|di-fgsm
- --pgd-iters, --pgd-step, --pgd-random-start
- --di-iters, --di-step, --di-prob, --di-scale-min, --di-scale-max, --di-random-start

## Outputs
- Adversarial images and visualizations are saved under ress/seg
- If segmentation is enabled, masks and overlays are saved alongside metrics CSV/XLSX files
