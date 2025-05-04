# 😊 Face Emotion Detection using FER-2013

This project is a machine learning-based application that detects human facial emotions from grayscale images using deep learning. It is built using the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), a widely used dataset for facial expression recognition.

## 🔍 Project Overview

The goal of this project is to classify facial expressions into one of the following three categories:

- **Happy**
- **Sad**
- **Neutral**

To improve classification accuracy, we designed a hybrid deep learning model that combines **Convolutional Neural Networks (CNN)** for feature extraction and a **Multilayer Perceptron (MLP)** for final emotion classification. This combination enhances both spatial feature understanding and dense layer decision-making.

## 📁 Dataset

- **Name**: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Format**: CSV
- **Images**: 48x48 grayscale
- **Total Samples**: 35,887 images
- **Split**: Training (28,709), Public Test (3,589), Private Test (3,589)

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (for preprocessing and visualization)

## 🧠 Model Architecture

- **CNN** layers: Extract spatial features from face images
- **Flatten** layer: Converts the 2D feature maps into a 1D vector
- **MLP** layers: Perform classification based on extracted features
- **Activation**: ReLU for hidden layers, Softmax for output

## 📊 Model Performance

After training the hybrid CNN-MLP model, we achieved improved accuracy over a standalone CNN, showing that combining deep feature extraction with dense decision layers can enhance emotion recognition performance.

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/AI-bootcamp/computer-vision-week-project-mimotion.git
