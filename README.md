# 🧠 Breast Cancer Classification with Neural Network & Streamlit

This project classifies breast tumors as **malignant** or **benign** using a trained neural network on the **Breast Cancer Wisconsin dataset**. It includes both a **Jupyter Notebook** for data processing and model training, and a **Streamlit app** for interactive predictions.

---

## 📌 Overview

- 📊 Dataset: `sklearn.datasets.load_breast_cancer()`
- 🧠 Model: Neural Network built with **TensorFlow/Keras**
- 🔍 Features: 30 numeric features extracted from digitized tumor images
- 🎯 Output: Predicts if the tumor is **Benign (0)** or **Malignant (1)**
- 💻 Frontend: Built with **Streamlit** for real-time input and prediction

---

## 🧠 Model Architecture

- Input layer: Flattened vector of 30 features  
- Hidden layer: 1 Dense layer with 20 ReLU units  
- Output layer: 2 neurons with Sigmoid activation  
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `Adam`

---

## 📂 Project Structure
