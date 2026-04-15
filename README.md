# 🧠 Brain Tumor Classification using Deep Learning and Explainable AI (XAI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-DL-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> A deep learning pipeline for classifying brain MRI scans into **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**, enhanced with **Explainable AI (Grad-CAM)** to provide transparent, clinically interpretable predictions.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
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

## 🔍 Overview

Brain tumors account for **85%–90% of all primary central nervous system (CNS) tumors** worldwide, and early, accurate diagnosis is critical for treatment planning and patient survival. Manual interpretation of MRI scans is time-consuming, subject to inter-observer variability, and requires specialist expertise.

This project presents an end-to-end deep learning solution that:

1. **Automatically classifies** brain MRI scans into four categories.
2. **Preprocesses and augments** images to improve generalization.
3. Uses **Grad-CAM** to visually explain *why* the model made a particular prediction — a crucial feature for real-world clinical adoption.

---

## 🎯 Motivation

While deep learning models achieve impressive performance on medical imaging tasks, their "black-box" nature limits clinical trust and adoption. This project bridges that gap by combining **high-accuracy classification** with **explainability**, enabling radiologists to:

- Verify that the model focuses on medically relevant regions.
- Detect potential biases or spurious correlations.
- Build confidence in AI-assisted diagnostic workflows.

---

## 📊 Dataset

The model is trained on a publicly available **Brain MRI dataset** from Kaggle, containing labeled MRI scans across four classes:

| Class | Description |
|---|---|
| **Glioma** | Tumors originating in the brain and/or spinal cord (e.g., astrocytomas, ependymomas, oligodendrogliomas). |
| **Meningioma** | Tumors arising from the meninges — typically slow-growing and often benign. |
| **Pituitary Tumor** | Tumors forming in the pituitary gland; most are benign adenomas. |
| **No Tumor** | Healthy brain MRI scans used as the control class. |

### Preprocessing Pipeline
- **Cropping** to remove irrelevant black borders and focus on the brain region.
- **Resizing** to a standard input shape.
- **Normalization** of pixel values to `[0, 1]`.
- **Data augmentation** (rotation, zoom, flipping) to improve robustness.

---

## 🏗 Model Architecture

The classifier is built using a **Convolutional Neural Network (CNN)** implemented in Keras/TensorFlow. The pipeline consists of:

1. **Input layer** — accepts preprocessed MRI images.
2. **Convolutional + pooling blocks** — extract hierarchical spatial features.
3. **Dense layers** with dropout for regularization.
4. **Softmax output layer** — produces class probabilities over the four tumor categories.

Training uses categorical cross-entropy loss with the Adam optimizer and early stopping on validation loss.

---

## 🔬 Explainable AI (XAI)

To make predictions interpretable, this project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which highlights the regions of the MRI scan that most influenced the model's decision.

**Benefits:**
- Visual heatmaps overlaid on the original scan.
- Confirms the network attends to tumor regions rather than artifacts.
- Supports radiologist review and model debugging.

---

## 📁 Project Structure

```
Brain-Tumor-classification-problem-using-deep-learning-and-XAI/
├── Brat.ipynb                                          # Main Jupyter notebook (end-to-end pipeline)
├── Brain Tumor classification498R-Project-Report-main2.pdf   # Full project report
├── brat-1.pdf                                          # Supporting documentation
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
pip install numpy pandas matplotlib seaborn opencv-python tensorflow keras tqdm scikit-learn jupyter
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
1. **Load & explore** the dataset.
2. **Preprocess** (crop, resize, normalize, augment).
3. **Build & compile** the CNN.
4. **Train** with validation monitoring.
5. **Evaluate** on the test set (accuracy, confusion matrix, classification report).
6. **Generate Grad-CAM** visualizations on sample predictions.

---

## 📈 Results

The trained model successfully classifies brain MRI scans across all four categories and produces interpretable Grad-CAM heatmaps highlighting tumor regions.

> 📄 For detailed metrics (accuracy, precision, recall, F1-score, confusion matrix, and training curves), refer to the full project report:
> [`Brain Tumor classification498R-Project-Report-main2.pdf`](./Brain%20Tumor%20classification498R-Project-Report-main2.pdf)

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | NumPy, Pandas, OpenCV (cv2) |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Grad-CAM |
| **Environment** | Jupyter Notebook, Google Colab |
| **Utilities** | tqdm, scikit-learn |

---

## 🗺 Roadmap

- [ ] Integrate transfer learning with pretrained backbones (VGG16, ResNet50, EfficientNet).
- [ ] Add additional XAI methods (LIME, SHAP, Integrated Gradients).
- [ ] Deploy as a web application (Streamlit / Flask / FastAPI).
- [ ] Expand to multi-modal MRI inputs (T1, T2, FLAIR).
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
- Authors of the original Grad-CAM paper: *Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, ICCV 2017.
- Open-source contributors behind TensorFlow, Keras, and the broader Python scientific stack.
- Prior work and inspiration from similar brain tumor classification repositories.

---

## 📬 Contact

**Author:** [kawserutshopro321](https://github.com/kawserutshopro321)

**Project Repository:** [Brain-Tumor-classification-problem-using-deep-learning-and-XAI](https://github.com/kawserutshopro321/Brain-Tumor-classification-problem-using-deep-learning-and-XAI)

For questions, suggestions, or collaborations, feel free to open an [issue](https://github.com/kawserutshopro321/Brain-Tumor-classification-problem-using-deep-learning-and-XAI/issues) or reach out via GitHub.

---

> ⚠️ **Disclaimer:** This project is for **research and educational purposes only**. It is **not** intended for clinical diagnosis or as a substitute for professional medical advice. Always consult qualified healthcare professionals for medical decisions.
## Results<a name="results"></a>

In the end, I could validate a test image passed through the model.

![validation](https://github.com/nazianafis/Brain-Tumor-Classification/blob/main/screenshots/valid-img.png)
