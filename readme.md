# FantasyIDiap ICCV25 Challenge - Image DeepFake Detection and Binary Classification 

This repository contained a complete pipeline for the **FantasyIDiap ICCV25 Challenge**, which focused on distinguishing between **bonafide** (genuine) and **attack** (forged) identity card images. The project utilized multiple deep learning models, including **ResNet18**, **ResNet50**, and **Swin Transformers**, to address the critical issue of identity document forgery in various digital applications.

---

## 📌 Problem Statement

In the modern digital landscape, identity cards served essential purposes such as **KYC verifications**, **financial onboarding**, and **authentication**. However, **identity card forgery** emerged as a widespread fraudulent practice, involving manipulation of elements such as **ID numbers**, **names**, **dates of birth**, and **photographs**.

This project aimed to develop an **AI-based model** that automatically classified identity documents as **genuine or fake** by analyzing their **visual features** and **structural patterns**. The challenge was addressed using the **FantasyID Dataset**, a **synthetic yet realistic** dataset specifically designed for forgery detection.

---

## 📂 Dataset Overview

**Classes of Images:**

* **bonafide**: Genuine ID cards
* **digital\_1**: Digitally forged copies (attack type 1)
* **digital\_2**: Digitally forged copies with alternate transformations (attack type 2)

**Multiple Device Captures:**

* Huawei
* iPhone 15 Pro
* Scanner

---

### 📁 Dataset Structure

```
FantasyIDiap-ICCV25-Challenge/
├── attack/
│   ├── digital_1/
│   │   ├── huawei/
│   │   ├── iphone15pro/
│   │   └── scan/
│   └── digital_2/
│       ├── huawei/
│       ├── iphone15pro/
│       └── scan/
├── bonafide/
│   ├── huawei/
│   ├── iphone15pro/
│   └── scan/
├── fantasyIDiap-train.csv
├── fantasyIDiap-test.csv
├── fantasyIDiap-oneset.csv
```

---

## 📊 Dataset Characteristics

* **Training Set**: 1,899 images (633 bonafide, 1,266 attacks)
* **Test Set**: 459 images (153 bonafide, 306 attacks)
* Maintained an approximate **2:1 attack-to-bonafide ratio**
* Contained rich metadata fields: `path`, `label`, `device`, `attack_type`, `split`, `relative_path`

---

## ⚙️ Installation

Ran in the provided notebook:

```bash
pip install opencv-python seaborn tqdm torch torchvision torchaudio timm
```

---

## ▶️ Running the Notebook

1. Cloned the repository or downloaded the `.ipynb` file.
2. Set the dataset path:

```python
DATASET_PATH = "/path/to/FantasyIDiap-ICCV25-Challenge"
```

3. Uncommented a model:

* **ResNet18**
* **ResNet50**
* **Swin Transformer**

4. Executed the notebook cell-by-cell in **Jupyter**, **VSCode**, or **Google Colab**.

---

## 🚀 Features

* Performed visual inspection of images
* Prepared and analyzed metadata
* Conducted data integrity checks
* Set up **PyTorch Datasets & Dataloaders**
* Supported multiple model options (**ResNet18**, **ResNet50**, **Swin Transformer**)
* Trained models per device & per attack type
* Generated accuracy metrics, confusion matrices, and performance logs

---

## 🧪 Methodology

### **General Data Preprocessing & Augmentation**

* Resized images to **224×224**
* Applied random horizontal flip (p=0.5)
* Added color jittering
* Used random affine transformations
* Normalized images to `[-1, 1]`
* Employed **WeightedRandomSampler** for class balance

---

### **Model-Specific Training**

#### **ResNet18**

* Utilized a pre-trained ImageNet model
* Replaced final layer with: Dropout(0.4) → Linear(512→2)
* Fine-tuned entire backbone after initial freezing
* Optimizer: Adam (`lr=1e-4`)
* Loss: CrossEntropyLoss

#### **ResNet50**

* Unfroze `layer4` + classifier head
* Custom head: BatchNorm → Dropout(0.5) → Linear(256) → ReLU → BatchNorm → Dropout(0.5) → Linear(2)
* Optimizer: AdamW (`lr=1e-4`)
* Loss: CrossEntropyLoss (weighted)
* Trained for 17 epochs

#### **Swin Transformer**

* Used Tiny variant pre-trained with `Swin_T_Weights.DEFAULT`
* Replaced head with Linear(in\_features→2)
* Trained last 40% of backbone
* Loss: Label Smoothing Cross-Entropy (`smoothing=0.1`)
* Scheduler: CosineAnnealingLR (10 epochs)

---

## 📊 Evaluation Strategy & Metrics

* Evaluated 12 trained model combinations on **6 custom test configurations**
* **Metrics**: Accuracy, AUC, Confusion Matrix
* **Visualizations**:

  * Accuracy heatmaps
  * ROC curves
  * Grad-CAM visualizations
  * Diagnostic plots
  * Boxplots & bar charts for devices & attack types

---

## 📊 Results

* **Highlights**:

  * Heatmaps revealed cross-device & cross-attack generalization
  * Confusion matrices showed misclassification patterns
  * Grad-CAM visualizations highlighted focus areas
  * Boxplots identified hardest device/attack combinations

---

## 🔄 Re-Evaluation

Reloaded saved models & switched architectures for benchmarking.

---

## ✅ Built-in Checks

* Validated image loading
* Raised errors for missing labels
* Skipped empty batches
* Handled empty splits
* Fixed `get_predictions()` typo
* Ensured reproducibility with `set_seed()`

---

## 📦 Requirements

* Python ≥ 3.7
* PyTorch ≥ 1.12
* torchvision
* timm
* tqdm
* OpenCV
* matplotlib
* seaborn
* pandas
* numpy

---

## 📬 Questions?

Opened an issue or contacted the maintainers.
**Enjoyed experimenting with CNNs and Transformers! 🤖**
