# 🧠 Brain Tumor Classification using Dense EfficientNet and Explainable AI (XAI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-DL-red.svg)](https://keras.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen.svg)](#-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> A CNN-based **Dense EfficientNet** pipeline for classifying **3,260 T1-weighted contrast-enhanced brain MRI scans** into **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** — enhanced with **Grad-CAM** Explainable AI for transparent, clinically interpretable predictions.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Key Highlights](#-key-highlights)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Explainable AI (XAI)](#-explainable-ai-xai)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

## 🔍 Problem Statement

Brain tumors are **most common in children and the elderly** and represent a serious form of cancer caused by uncontrollable brain cell growth inside the skull. Tumor cells are notoriously difficult to classify due to their **heterogeneity** — they vary significantly in shape, location, texture, and intensity on MRI scans.

Manual interpretation of MRI scans is time-consuming, subject to inter-observer variability, and requires specialist expertise. Convolutional Neural Networks (CNNs) are the most widely used deep learning algorithm for visual learning and brain tumor recognition, making them a natural fit for automating this diagnostic task.

---

## 💡 Proposed Solution

This study proposes a **CNN-based Dense EfficientNet** for classifying **3,260 T1-weighted contrast-enhanced brain MRI images** into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The model accurately categorizes a limited database, achieving:

- ✅ **99.96% accuracy** during training
- ✅ **100% accuracy** during testing

The **Grad-CAM** technique is used to identify the key image features driving each classification — making the model's decisions transparent and clinically verifiable.

---

## ✨ Key Highlights

| Feature | Detail |
|---|---|
| 🧬 **Model** | CNN-based **Dense EfficientNet** |
| 🖼 **Dataset Size** | **3,260** T1-weighted contrast-enhanced brain MRIs |
| 🏷 **Classes** | Glioma · Meningioma · Pituitary · No Tumor |
| 🎯 **Training Accuracy** | **99.96%** |
| 🎯 **Testing Accuracy** | **100%** |
| 🔬 **Explainability** | **Grad-CAM** heatmaps highlight key features |
| 🧪 **Environment** | Python · TensorFlow / Keras · Jupyter / Colab |

---

## 🏛 Architecture

The diagram below shows the full pipeline — from 3,260 T1-weighted contrast-enhanced MRI inputs, through preprocessing and the Dense EfficientNet classifier, to the Grad-CAM explainability branch that produces interpretable heatmaps alongside the prediction.

<p align="center">
  <img src="architecture.svg" alt="Brain Tumor Classification — Dense EfficientNet + Grad-CAM Architecture" width="100%"/>
</p>

---

## 📊 Dataset

The model is trained on **3,260 T1-weighted contrast-enhanced brain MRI images**, labeled across four classes:

| Class | Description |
|---|---|
| **Glioma** | Tumors originating in the brain and/or spinal cord (e.g., astrocytomas, ependymomas, oligodendrogliomas). |
| **Meningioma** | Tumors arising from the meninges — typically slow-growing and often benign. |
| **Pituitary Tumor** | Tumors forming in the pituitary gland; most are benign adenomas. |
| **No Tumor** | Healthy brain MRI scans used as the control class. |

### Preprocessing Pipeline
- **Cropping** to remove irrelevant black borders and focus on the brain region.
- **Resizing** to a standard input shape suitable for EfficientNet.
- **Normalization** of pixel values.
- **Data augmentation** (rotation, zoom, flipping) to improve robustness on a limited database.

---

## 🏗 Model Architecture

The classifier is built as a **CNN-based Dense EfficientNet**, combining the compound-scaling efficiency of EfficientNet with densely connected feature reuse. The pipeline consists of:

1. **Input layer** — accepts preprocessed T1-weighted contrast-enhanced MRI images.
2. **EfficientNet MBConv blocks** — compound-scaled (depth × width × resolution) mobile inverted bottleneck convolutions extract hierarchical spatial features efficiently.
3. **Dense connections** — feature reuse across blocks improves gradient flow and representational capacity on the limited dataset.
4. **Classifier head** — Global Average Pooling → Dense → Dropout → Softmax over 4 classes.
<img width="1341" height="776" alt="image" src="https://github.com/user-attachments/assets/c2d322bc-8d2c-41be-ba68-c5943ea7fcec" />

Training uses categorical cross-entropy loss with the Adam optimizer, early stopping on validation loss, and aggressive augmentation to prevent overfitting on the 3,260-image dataset.

---

## 🔬 Explainable AI (XAI)

To make predictions interpretable, this project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which uses the gradients of the target class flowing into the final convolutional layer to produce a **class-discriminative heatmap** highlighting the image regions most responsible for the prediction.

**Benefits:**
- Identifies the **key features** driving each classification.
- Confirms the network attends to tumor regions rather than imaging artifacts.
- Supports radiologist review and clinical trust.
- Enables model debugging and bias detection.

---

## 📁 Project Structure

```
Brain-Tumor-classification-problem-using-deep-learning-and-XAI/
├── Brat.ipynb                                          # Main Jupyter notebook (end-to-end pipeline)
├── Brain Tumor classification498R-Project-Report-main2.pdf   # Full project report
├── brat-1.pdf                                          # Supporting documentation
├── architecture.svg                                    # Architecture diagram
├── LICENSE                                             # MIT license
└── README.md                                           # Project documentation
```

---

## ⚙️ Installation

### Prerequisites
- Python **3.8+**
- pip or conda
- (Recommended) GPU with CUDA support for faster training
- Or simply use **Google Colab** (no local setup required)

### 1. Clone the Repository
```bash
git clone https://github.com/kawserutshopro321/Brain-Tumor-classification-problem-using-deep-learning-and-XAI.git
cd Brain-Tumor-classification-problem-using-deep-learning-and-XAI
```

### 2. Create a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow keras efficientnet tqdm scikit-learn jupyter
```

> 💡 For Grad-CAM visualizations, `tf-keras-vis` or a custom Grad-CAM implementation is used inside the notebook.

---

## 🚀 Usage

### Option 1: Google Colab (Recommended)
1. Open `Brat.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload the Kaggle dataset or mount Google Drive.
3. Run all cells sequentially.

### Option 2: Local Jupyter Notebook
```bash
jupyter notebook Brat.ipynb
```

### Workflow Inside the Notebook
1. **Load & explore** the 3,260-image dataset.
2. **Preprocess** (crop, resize, normalize, augment).
3. **Build & compile** the Dense EfficientNet model.
4. **Train** with validation monitoring and early stopping.
5. **Evaluate** on the test set (accuracy, confusion matrix, classification report).
6. **Generate Grad-CAM** visualizations to identify key features.

---

## 📈 Results

The Dense EfficientNet model achieves **state-of-the-art performance** on the 3,260-image brain MRI dataset:

| Metric | Value |
|---|---|
| **Training Accuracy** | **99.96%** |
| **Testing Accuracy** | **100%** |
| **Classes** | 4 (Glioma, Meningioma, Pituitary, No Tumor) |
| **Explainability** | Grad-CAM heatmaps |

Grad-CAM visualizations confirm the model consistently attends to tumor-relevant regions, supporting the validity of its predictions.

> 📄 For detailed metrics (precision, recall, F1-score, confusion matrix, and training curves), refer to the full project report:
> [`Brain Tumor classification498R-Project-Report-main2.pdf`](./Brain%20Tumor%20classification498R-Project-Report-main2.pdf)

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Deep Learning** | TensorFlow, Keras, EfficientNet |
| **Data Processing** | NumPy, Pandas, OpenCV (cv2) |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Grad-CAM |
| **Environment** | Jupyter Notebook, Google Colab |
| **Utilities** | tqdm, scikit-learn |

---

## 🗺 Roadmap

- [ ] Compare against other pretrained backbones (VGG16, ResNet50, DenseNet, Vision Transformers).
- [ ] Add additional XAI methods (LIME, SHAP, Integrated Gradients).
- [ ] Deploy as a web application (Streamlit / Flask / FastAPI).
- [ ] Expand to multi-modal MRI inputs (T1, T2, FLAIR).
- [ ] Validate on larger and external datasets to test generalization.
- [ ] Publish a Dockerized version for reproducibility.
- [ ] Add unit tests and CI/CD pipeline.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes relevant documentation.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- The Kaggle community for providing the Brain MRI dataset.
- Authors of **EfficientNet**: *Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"*, ICML 2019.
- Authors of the original **Grad-CAM** paper: *Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, ICCV 2017.
- Open-source contributors behind TensorFlow, Keras, and the broader Python scientific stack.

---

## 📬 Contact

**Author:** [kawserutshopro321](https://github.com/kawserutshopro321)

**Project Repository:** [Brain-Tumor-classification-problem-using-deep-learning-and-XAI](https://github.com/kawserutshopro321/Brain-Tumor-classification-problem-using-deep-learning-and-XAI)

For questions, suggestions, or collaborations, feel free to open an [issue](https://github.com/kawserutshopro321/Brain-Tumor-classification-problem-using-deep-learning-and-XAI/issues) or reach out via GitHub.

---

> ⚠️ **Disclaimer:** This project is for **research and educational purposes only**. It is **not** intended for clinical diagnosis or as a substitute for professional medical advice. Always consult qualified healthcare professionals for medical decisions.
