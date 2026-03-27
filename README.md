# 🧠 ResistAI - Intelligent Antibiotic Resistance Predictor

## 🚀 Overview
ResistAI is an AI-powered decision support system designed to predict antibiotic resistance and recommend effective treatments. It helps healthcare professionals make faster and more accurate decisions using machine learning.

---

## 💡 Problem Statement
Antibiotic resistance is a growing global health issue. Incorrect or delayed treatment can lead to serious complications. There is a need for a smart system that can assist in predicting resistance patterns and suggesting suitable antibiotics.

---

## 🎯 Solution
ResistAI uses machine learning models trained on medical datasets to:
- Predict whether a bacteria is **Susceptible, Intermediate, or Resistant**
- Recommend the most effective antibiotics
- Provide data-driven insights for better decision-making

---

## ✨ Features
- 🔍 Antibiotic Resistance Prediction  
- 💊 Smart Recommendation System  
- 📊 Data Preprocessing & Cleaning  
- 🤖 Machine Learning Model Integration  
- 🌐 User-friendly Interface (Streamlit)  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Framework:** Streamlit  
- **Model Storage:** Pickle (`.pkl`)  

---
```
ResistAI/
│
├── app/
│   └── app.py                 # Main Streamlit application
│
├── data/
│   ├── primary.xlsx          # Original dataset 1
│   ├── secondary.csv         # Original dataset 2
│   └── cleaned_data.csv      # Processed dataset
│
├── models/
│   └── model.pkl             # Trained ML model
│
├── src/
│   ├── preprocess.py         # Data cleaning & preprocessing
│   ├── train.py              # Model training pipeline
│   ├── recommend.py          # Recommendation engine
│   └── visualize.py          # Charts & analytics
│
├── outputs/
│   └── metrics.json          # Model evaluation results
│
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .gitignore                # Ignore unnecessary files
```
