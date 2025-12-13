# BIA4-ICA1-Group3
This repository is for a project of ICA1 in BIA4 to solve the neuron dendritic spine classification by Group3 of ZJE students.

# Dendritic Spine Classification with TinyCNN

This repository provides a lightweight and efficient deep learning pipeline for **three-class dendritic spine classification** (Mushroom, Stubby, Thin) using a custom-designed **TinyCNN** architecture.

The project focuses on **practical deployment**, **computational efficiency**, and **robust performance**, demonstrating that compact convolutional networks can outperform larger models for small-scale biological image classification tasks.

---

## Key Features

*  **TinyCNN-based classifier** optimized for dendritic spine morphology
*  End-to-end pipeline: data split → training → evaluation → benchmarking
*  Fair benchmarking against **ViT-B16, ResNet18, CNN-3Layer, SVM, and Random Forest**
*  Fixed train/test split for reproducibility
*  Evaluation with **Accuracy, F1, AUC, and inference latency**

---

## Why TinyCNN?

We systematically benchmarked multiple models on the same dataset and found that **TinyCNN offers the best overall trade-off between performance and efficiency**.

| Model            | Params   | Accuracy   | F1         | AUC       | Inference (ms/image) |
| ---------------- | -------- | ---------- | ---------- | --------- | -------------------- |
| **TinyCNN**      | **457K** | **94.57%** | **92.71%** | **0.987** | **27.1**             |
| ViT-B16 (frozen) | 85.8M    | 95.65%     | 94.60%     | 0.991     | 95.6                 |
| ResNet18         | 11.3M    | 93.48%     | 90.29%     | 0.984     | 433.6                |
| CNN-3Layer       | 8.5M     | 90.22%     | 85.31%     | 0.968     | 15.8                 |
| Random Forest    | –        | 91.30%     | 88.29%     | 0.966     | 0.14                 |
| SVM              | –        | 82.61%     | 74.84%     | 0.867     | 6.70                 |

 **Conclusion**
While transformer-based models (ViT) achieve slightly higher accuracy, they are **computationally excessive** for a three-class task. TinyCNN achieves near-optimal performance using **<1% of ViT’s parameters**, making it the **recommended model for deployment and large-scale analysis**.

---

## Model Architecture: TinyCNN

```text
Input (RGB 250×250)
↓
Conv(32) → BN → ReLU → MaxPool
↓
Conv(64) → BN → ReLU → MaxPool
↓
Conv(128) → BN → ReLU → MaxPool
↓
Conv(256) → BN → ReLU → MaxPool
↓
AdaptiveAvgPool
↓
FC (256 → 192 → 96 → 3)
```

* Total parameters: **~457K**
* Designed specifically for **small morphological datasets**
* Avoids overfitting while preserving discriminative power

---

## Repository Structure

```text
.
├── software/
│   ├── tinycnn.py
│   ├── resnet18.py
│   ├── vit.py
│
├── data/
│   ├── spine_train_split.csv
│   ├── spine_test_split.csv
│
├── documentation/
│   ├── train_tinycnn.py
│   ├── train_resnet18.py
│
├── contributions/
│   ├── run_benchmark.py
│
├── weights/
│   ├── Spine_tinyCNN.pth
│   ├── Spine_ResNet18.pth
│   ├── Spine_ViT.pth
│
├── benchmark_results.csv
├── requirements.txt
└── README.md
```

---

## Dataset

* Input: RGB dendritic spine images
* Labels: `Mushroom`, `Stubby`, `Thin`
* Fixed train/test split stored as CSV for reproducibility
* Image size:

  * TinyCNN / ResNet18: **250 × 250**
  * ViT / CNN-3Layer: **224 × 224**

---

## Installation

```bash
git clone https://github.com/atrpy/BIA4-ICA1-Group3.git
cd dendritic-spine-tinycnn
pip install -r requirements.txt
```

---

## Training TinyCNN

```bash
python training/train_tinycnn.py
```

Best model checkpoints are saved automatically based on validation accuracy.

---

## Benchmarking

Run all models on the same test set:

```bash
python benchmark/run_benchmark.py
```

This script reports:

* Accuracy
* F1
* AUC
* Inference latency
* Parameter count

Results are saved to `benchmark_results.csv`.

---

## Intended Use

* Automated dendritic spine morphology analysis
* Large-scale neuroscience image datasets
* Resource-constrained environments (edge devices, lab servers)
* Rapid prototyping and deployment

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tinycnn_spine,
  title = {Efficient Dendritic Spine Classification with TinyCNN},
  author = {Your Name},
  year = {2025},
}
```

---

## Acknowledgements

This project was developed as part of a computational neuroscience and biomedical image analysis workflow, emphasizing **model efficiency**, **reproducibility**, and **biological interpretability**.

---
