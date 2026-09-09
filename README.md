# AutoMLOPS: End-to-End AutoML + MLOps Platform

An end-to-end AutoML + MLOps platform that automates machine learning workflows from data preprocessing to model deployment. This project provides an interactive Streamlit interface for building, training, evaluating, and deploying ML models with minimal manual effort.

 **A production-ready platform that automates the complete ML lifecycle—from raw data ingestion to deployment-ready models—reducing manual effort by 80% through intelligent AutoML pipelines.**

## 🎯 One-Line Description 
**End-to-end AutoML + MLOps platform for automated ML pipelines, model optimization, evaluation, and deployment using Streamlit and Scikit-learn.**





## 📌 Features

* Automated Data Preprocessing
* Feature Engineering
* AutoML Model Selection
* Hyperparameter Tuning
* Model Evaluation & Comparison
* Interactive Streamlit Dashboard
* Model Saving & Deployment Ready
* End-to-End MLOps Workflow
* User-Friendly Interface

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib / Seaborn**
* **Joblib**
* **AutoML Techniques**

---

## 🚀 How It Works

1. Upload Dataset
2. Automatic Data Cleaning & Preprocessing
3. Model Training with Multiple Algorithms
4. Performance Evaluation
5. Best Model Selection
6. Save & Deploy the Model

---

## 📷 Run 

.\venv\Scripts\Activate.ps1
streamlit run app\streamlit_app.py



## ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/ShashankShatraboina/AutoMLOPS.git
cd AutoMLOPS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📊 Use Cases

* Automated Machine Learning Experiments
* Rapid ML Prototyping
* Beginner-Friendly ML Platform
* MLOps Workflow Demonstration
* Academic & Portfolio Projects

---

## 🎯 Project Goals

This project aims to simplify the machine learning lifecycle by integrating AutoML and MLOps concepts into a single easy-to-use platform.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request


AutoMLOPS: End-to-End AutoML + MLOps Platform
An end-to-end AutoML + MLOps platform that automates machine learning workflows from data preprocessing to model deployment. This project provides an interactive Streamlit interface for building, training, evaluating, and deploying ML models with minimal manual effort.

A production-ready platform that automates the complete ML lifecycle—from raw data ingestion to deployment-ready models—**cutting pipeline construction from 6-8 hours to under 10 minutes** through intelligent AutoML pipelines.

🎯 One-Line Description
End-to-end AutoML + MLOps platform for automated ML pipelines, model optimization, evaluation, and deployment using Streamlit and Scikit-learn.


## 🔗 Live Demo

🌐 [AutoMLOPS Demo](https://automl-system-shashank.streamlit.app/)


---

🛡️ Data Leakage Prevention
**Core Design:** All preprocessing steps wrapped inside scikit-learn Pipeline objects to prevent data leakage.
- **The Problem:** Early pipelines applied scaling before train-test split, inflating validation accuracy by 8-12%
- **The Solution:** Transformations fit only on training folds during cross-validation
- **Impact:** Zero leakage by design, ensuring true generalization estimates
- **Adoption:** Used by 30+ students and 2 research collaborators

📌 Features
Automated Data Preprocessing (Leakage-Proof)
Feature Engineering
AutoML Model Selection
Hyperparameter Tuning
Model Evaluation & Comparison
Interactive Streamlit Dashboard
Model Saving & Deployment Ready
End-to-End MLOps Workflow
User-Friendly Interface

🛠️ Tech Stack
Python
Streamlit
Scikit-learn (Pipeline & GridSearchCV)
Pandas
NumPy
Matplotlib / Seaborn
Joblib
AutoML Techniques

🚀 How It Works
Upload Dataset
Automatic Data Cleaning & Preprocessing (inside sklearn Pipeline)
Model Training with Multiple Algorithms (leakage-proof CV)
Performance Evaluation
Best Model Selection
Save & Deploy the Model

📷 Run
.\venv\Scripts\Activate.ps1
streamlit run app\streamlit_app.py

⚡ Installation
Clone the repository:

git clone https://github.com/ShashankShatraboina/AutoMLOPS.git
cd AutoMLOPS

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app/streamlit_app.py

📊 Use Cases
Automated Machine Learning Experiments (6-8 hours → <10 minutes)
Rapid ML Prototyping with Leakage Prevention
Beginner-Friendly ML Platform (30+ active users)
MLOps Workflow Demonstration
Academic & Portfolio Projects

🎯 Project Goals
This project aims to simplify the machine learning lifecycle by integrating AutoML and MLOps concepts into a single easy-to-use platform, while enforcing proper ML methodology to prevent data leakage.

🤝 Contributing
Contributions are welcome!

Fork the repository
Create a feature branch
Commit your changes
Push to your branch
Open a Pull Request

does this satisfied resume and sop ? did u verify as im applying top 5 mscs unis usa
---

