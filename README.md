# BIA4-ICA1-Group3

This repository contains the Group 3 project for **ICA1 in BIA4**, developed by students from the **Zhejiang University – University of Edinburgh Institute (ZJE)**.

This project aims to provide a **practical and efficient solution for large-scale dendritic spine classification**, supporting neuroscience researchers in the **rapid, batch-wise analysis** of neuronal dendritic spine morphology.

---

# SpineClassifier: Dendritic Spine Classification with TinyCNN

This repository presents a lightweight yet robust deep learning pipeline for **three-class dendritic spine classification** (*Mushroom, Stubby, Thin*) based on a custom-designed **TinyCNN** architecture.

The software is designed to address a common bottleneck in neuroscience research:
**the need for fast, reliable, and scalable classification of dendritic spine morphologies from microscopy images**.

By emphasizing **computational efficiency, reproducibility, and practical usability**, this project demonstrates that compact convolutional neural networks can outperform or closely match large-scale models on small-to-medium biological image datasets.

---

## ⚠️ IMPORTANT – How to Use the Software "SpineClassifier"

**The officially usable software (GUI-based application) is provided ONLY in the GitHub Releases section.**

**Please download the appropriate version (Windows or macOS) from the Release page and follow the usage video in the `documentation/` folder before running the software.**

The code in this repository is primarily intended for **development, training, benchmarking, and academic inspection**, rather than direct end-user execution.

---
## macOS Security Notice

When downloading the macOS version of this software from the internet, macOS may display a warning stating that the application cannot be opened because it is from an unidentified developer.

To resolve this issue:

1. Open **System Settings**
2. Navigate to **Privacy & Security**
3. Scroll down to the **Security** section
4. Click **Allow Anyway** next to the blocked application
5. Re-open the application to proceed

This is a standard macOS security mechanism for applications distributed outside the App Store and does not indicate malicious behavior.

## Key Features

* TinyCNN-based classifier optimized for dendritic spine morphology
* End-to-end workflow: preprocessing → training → evaluation → benchmarking
* Fair benchmarking against **ViT-B16, ResNet18, CNN-3Layer, SVM, and Random Forest**
* Fixed train/test split stored as CSV files for reproducibility
* Evaluation using **Accuracy, F1-score, AUC, and inference latency**
* GUI-based software for non-programming users (via Release)

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

### Conclusion

Although transformer-based models such as ViT achieve slightly higher peak accuracy, they are **computationally excessive** for a three-class dendritic spine classification task.

TinyCNN achieves near-optimal performance using **less than 1% of the parameters of ViT**, while maintaining stable inference speed and strong generalization.
Therefore, **TinyCNN is the recommended model for real-world deployment and large-scale analysis** in neuroscience research settings.

---

## Model Architecture: TinyCNN

```text
Input (RGB 250×250)
↓
Conv(32) → BatchNorm → ReLU → MaxPool
↓
Conv(64) → BatchNorm → ReLU → MaxPool
↓
Conv(128) → BatchNorm → ReLU → MaxPool
↓
Conv(256) → BatchNorm → ReLU → MaxPool
↓
AdaptiveAvgPool
↓
Fully Connected (256 → 192 → 96 → 3)
```

* Total parameters: ~457K
* Specifically designed for **small-scale morphological datasets**
* Effectively balances feature representation and overfitting risk

---

## Repository Structure

```text
.
├── data/
│   ├── data_binary/
│   ├── data_intensity_3/
│
├── preprocessing/
│   ├── binary_v2.py
│   ├── readme.docx
│   ├── README_auto_binary.md
│   ├── auto_binary.py
│
├── model/
│   ├── RF_Group.ipynb
│   ├── SVM_Group.ipynb
│   ├── Resnet18.ipynb
│   ├── 3layer_CNN.py
│   ├── tinyCNN.ipynb
│   ├── ViT.ipynb
│
├── benchmark/
│   ├── benchmark.ipynb
│   ├── benchmark_results.csv
│   ├── infer_time.png
│   ├── paras.png
│   ├── acc f1&auc.png
│
├── documentation/
│   ├── documentation_video.mp4
│
├── contributions/
│   ├── contributions.docx
│
└── README.md

Release
├── SpineClassfier_for_mac/
│   ├── spine_mac.zip/
│   
├── SpineClassfier_for_Windows/
│   ├── Spine_APP.zip/

```

---

## Dataset

The training and evaluation process in this project **references and builds upon** the following publicly available dataset:

**Dendritic Spine Analysis Dataset**
[https://github.com/mughanibu/Dendritic-Spine-Analysis-Dataset](https://github.com/mughanibu/Dendritic-Spine-Analysis-Dataset)

In this repository:

* The `data/` folder contains two subfolders:

  * `data_binary/`
  * `data_intensity_3/`
* All images in these folders originate from the above dataset
* Images are reorganized and preprocessed to support standardized training and benchmarking

---

## Preprocessing Recommendation

For best performance, we **strongly recommend preprocessing raw microscopy images** before model inference.

After acquiring dendritic spine images, users should:

1. Follow the readme.docx provided in the `preprocessing/` folder
2. Convert raw images into **spine-centered, binary representations**
3. Ensure consistent image scale and background removal

This preprocessing step significantly improves classification robustness and aligns input data with the training distribution of TinyCNN.

---

## Benchmarking

All models are evaluated on the **same fixed test set** to ensure fairness.

```bash
python benchmark/benchmark.ipynb
```

Metrics reported:

* Accuracy
* F1-score
* Multi-class AUC
* Inference latency
* Parameter count

Results are saved to `benchmark_results.csv`.

---

## Intended Use

* High-throughput dendritic spine morphology analysis
* Neuroscience research requiring batch image processing
* Resource-constrained laboratory environments
* Rapid prototyping and methodological comparison

---

## Acknowledgements

We would like to sincerely thank the creators and maintainers of the Dendritic Spine Analysis Dataset
(https://github.com/mughanibu/Dendritic-Spine-Analysis-Dataset
) for making their dataset publicly available, which provided essential support for model training, evaluation, and benchmarking in this project.

We also thank all members of BIA4 ICA1 Group 3 for their collaborative effort, technical contributions, and constructive discussions throughout the development of this work.

In addition, we gratefully acknowledge the Bioimaging Analysis 4 (BIA4) course for providing the academic framework and technical guidance that motivated and shaped this project.

Finally, we thank the Zhejiang University – University of Edinburgh Institute (ZJE) for fostering an interdisciplinary learning environment that enabled the integration of computational methods with biological image analysis.
