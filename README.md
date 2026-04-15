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

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 620" font-family="Segoe UI, Arial, sans-serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#eef2f7"/>
    </linearGradient>
    <linearGradient id="input" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#60a5fa"/>
      <stop offset="100%" stop-color="#2563eb"/>
    </linearGradient>
    <linearGradient id="prep" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>
    <linearGradient id="conv" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#7c3aed"/>
    </linearGradient>
    <linearGradient id="dense" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f472b6"/>
      <stop offset="100%" stop-color="#db2777"/>
    </linearGradient>
    <linearGradient id="out" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fb923c"/>
      <stop offset="100%" stop-color="#ea580c"/>
    </linearGradient>
    <linearGradient id="xai" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f87171"/>
      <stop offset="100%" stop-color="#b91c1c"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#475569"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.15"/>
    </filter>
  </defs>

  <rect width="1200" height="620" fill="url(#bg)"/>

  <!-- Title -->
  <text x="600" y="40" text-anchor="middle" font-size="24" font-weight="700" fill="#0f172a">
    Brain Tumor Classification — Deep Learning + XAI Pipeline
  </text>
  <text x="600" y="65" text-anchor="middle" font-size="13" fill="#475569">
    End-to-end architecture from MRI input to explainable prediction
  </text>

  <!-- Stage 1: Input -->
  <g filter="url(#shadow)">
    <rect x="40" y="130" width="160" height="120" rx="12" fill="url(#input)"/>
    <text x="120" y="165" text-anchor="middle" font-size="15" font-weight="700" fill="white">MRI Input</text>
    <text x="120" y="190" text-anchor="middle" font-size="11" fill="#e0e7ff">Brain MRI Scan</text>
    <text x="120" y="210" text-anchor="middle" font-size="11" fill="#e0e7ff">(Kaggle Dataset)</text>
    <text x="120" y="232" text-anchor="middle" font-size="10" fill="#c7d2fe">4 classes</text>
  </g>

  <!-- Stage 2: Preprocessing -->
  <g filter="url(#shadow)">
    <rect x="240" y="130" width="180" height="120" rx="12" fill="url(#prep)"/>
    <text x="330" y="160" text-anchor="middle" font-size="15" font-weight="700" fill="white">Preprocessing</text>
    <text x="330" y="182" text-anchor="middle" font-size="11" fill="#d1fae5">• Crop brain region</text>
    <text x="330" y="200" text-anchor="middle" font-size="11" fill="#d1fae5">• Resize + Normalize</text>
    <text x="330" y="218" text-anchor="middle" font-size="11" fill="#d1fae5">• Augmentation</text>
    <text x="330" y="236" text-anchor="middle" font-size="11" fill="#d1fae5">  (rotate, flip, zoom)</text>
  </g>

  <!-- Stage 3: CNN Feature Extractor -->
  <g filter="url(#shadow)">
    <rect x="460" y="110" width="320" height="160" rx="12" fill="url(#conv)"/>
    <text x="620" y="138" text-anchor="middle" font-size="15" font-weight="700" fill="white">CNN Feature Extractor</text>

    <!-- Mini conv blocks -->
    <rect x="480" y="160" width="50" height="70" rx="4" fill="#ffffff" opacity="0.25"/>
    <text x="505" y="195" text-anchor="middle" font-size="10" font-weight="600" fill="white">Conv</text>
    <text x="505" y="210" text-anchor="middle" font-size="10" fill="white">+Pool</text>

    <rect x="545" y="168" width="50" height="62" rx="4" fill="#ffffff" opacity="0.28"/>
    <text x="570" y="200" text-anchor="middle" font-size="10" font-weight="600" fill="white">Conv</text>
    <text x="570" y="215" text-anchor="middle" font-size="10" fill="white">+Pool</text>

    <rect x="610" y="176" width="50" height="54" rx="4" fill="#ffffff" opacity="0.32"/>
    <text x="635" y="205" text-anchor="middle" font-size="10" font-weight="600" fill="white">Conv</text>
    <text x="635" y="220" text-anchor="middle" font-size="10" fill="white">+Pool</text>

    <rect x="675" y="184" width="50" height="46" rx="4" fill="#ffffff" opacity="0.36"/>
    <text x="700" y="210" text-anchor="middle" font-size="10" font-weight="600" fill="white">Conv</text>
    <text x="700" y="223" text-anchor="middle" font-size="9" fill="white">+Pool</text>

    <text x="620" y="255" text-anchor="middle" font-size="11" fill="#ede9fe">ReLU activations • Batch Norm • Dropout</text>
  </g>

  <!-- Stage 4: Dense -->
  <g filter="url(#shadow)">
    <rect x="820" y="130" width="160" height="120" rx="12" fill="url(#dense)"/>
    <text x="900" y="160" text-anchor="middle" font-size="15" font-weight="700" fill="white">Fully Connected</text>
    <text x="900" y="185" text-anchor="middle" font-size="11" fill="#fce7f3">Flatten → Dense</text>
    <text x="900" y="203" text-anchor="middle" font-size="11" fill="#fce7f3">Dropout</text>
    <text x="900" y="225" text-anchor="middle" font-size="11" fill="#fce7f3">Softmax (4 classes)</text>
  </g>

  <!-- Stage 5: Output -->
  <g filter="url(#shadow)">
    <rect x="1020" y="130" width="150" height="120" rx="12" fill="url(#out)"/>
    <text x="1095" y="158" text-anchor="middle" font-size="14" font-weight="700" fill="white">Prediction</text>
    <text x="1095" y="182" text-anchor="middle" font-size="10" fill="#ffedd5">• Glioma</text>
    <text x="1095" y="198" text-anchor="middle" font-size="10" fill="#ffedd5">• Meningioma</text>
    <text x="1095" y="214" text-anchor="middle" font-size="10" fill="#ffedd5">• Pituitary</text>
    <text x="1095" y="230" text-anchor="middle" font-size="10" fill="#ffedd5">• No Tumor</text>
  </g>

  <!-- Arrows (top row) -->
  <line x1="200" y1="190" x2="238" y2="190" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="420" y1="190" x2="458" y2="190" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="780" y1="190" x2="818" y2="190" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="980" y1="190" x2="1018" y2="190" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- XAI branch -->
  <g filter="url(#shadow)">
    <rect x="460" y="380" width="320" height="140" rx="12" fill="url(#xai)"/>
    <text x="620" y="410" text-anchor="middle" font-size="15" font-weight="700" fill="white">Explainable AI — Grad-CAM</text>
    <text x="620" y="435" text-anchor="middle" font-size="11" fill="#fee2e2">Gradients from target class →</text>
    <text x="620" y="453" text-anchor="middle" font-size="11" fill="#fee2e2">last conv layer activations</text>
    <text x="620" y="478" text-anchor="middle" font-size="11" fill="#fee2e2">Weighted heatmap overlaid on MRI</text>
    <text x="620" y="500" text-anchor="middle" font-size="11" fill="#fee2e2">→ shows <tspan font-weight="700">where</tspan> the model looked</text>
  </g>

  <!-- Heatmap output -->
  <g filter="url(#shadow)">
    <rect x="820" y="390" width="160" height="120" rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="900" y="418" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">Heatmap Overlay</text>
    <!-- mini heatmap illustration -->
    <circle cx="900" cy="460" r="32" fill="#fde68a"/>
    <circle cx="900" cy="460" r="22" fill="#fb923c"/>
    <circle cx="900" cy="460" r="12" fill="#dc2626"/>
    <text x="900" y="498" text-anchor="middle" font-size="10" fill="#475569">Interpretable for clinicians</text>
  </g>

  <!-- Arrow from CNN to Grad-CAM -->
  <path d="M 620 270 L 620 380" stroke="#b91c1c" stroke-width="2" stroke-dasharray="5,4" fill="none" marker-end="url(#arrow)"/>
  <text x="635" y="330" font-size="11" fill="#b91c1c" font-style="italic">activations +</text>
  <text x="635" y="345" font-size="11" fill="#b91c1c" font-style="italic">gradients</text>

  <!-- Arrow Grad-CAM → Heatmap -->
  <line x1="780" y1="450" x2="818" y2="450" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Legend / bottom caption -->
  <rect x="40" y="400" width="380" height="140" rx="12" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="60" y="428" font-size="13" font-weight="700" fill="#0f172a">Pipeline Summary</text>
  <text x="60" y="452" font-size="11" fill="#334155">1. MRI scans are cropped, normalized, and augmented.</text>
  <text x="60" y="470" font-size="11" fill="#334155">2. A CNN learns hierarchical spatial features.</text>
  <text x="60" y="488" font-size="11" fill="#334155">3. Dense layers produce a 4-class softmax prediction.</text>
  <text x="60" y="506" font-size="11" fill="#334155">4. Grad-CAM highlights image regions driving the</text>
  <text x="60" y="522" font-size="11" fill="#334155">    decision — enabling clinical interpretability.</text>

  <!-- Footer -->
  <text x="600" y="595" text-anchor="middle" font-size="10" fill="#94a3b8">
    Brain Tumor Classification using Deep Learning and XAI • Architecture Overview
  </text>
</svg>
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
