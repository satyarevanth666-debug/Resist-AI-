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

ResistAI/
|
│
├── app/                  
│   └── app.py            # Main Streamlit app
│
├── data/                 
│   ├── primary.xlsx      
│   ├── secondary.csv     
│   └── cleaned_data.csv  
│
├── models/               
│   └── model.pkl         
│
├── src/                  
│   ├── preprocess.py     
│   ├── train.py          
│   ├── recommend.py      
│   └── visualize.py      
│
├── outputs/              
│   └── metrics.json      
│
├── requirements.txt      
├── README.md             
└── .gitignore
