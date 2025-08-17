# Credit Card Churn Prediction  

## 1. Description  
This is an **end-to-end machine learning project** that applies **LightGBM** to predict a customer’s probability of churning in a bank’s credit card service.  
The project uses **supervised learning** for classification, where the target is **1 if the customer churned, else 0**.  

The workflow was designed following **CI/CD principles** and modular coding practices:  
- Initial analysis (EDA to modeling) was done in **Jupyter Notebooks**.  
- The project was modularized into components for **data ingestion, transformation, and model training**.  
- Automated **training and prediction pipelines** were built, with scripts producing model artifacts and consuming them for predictions.  
- Implemented **best practices**: virtual environments, exception handling, logging, and documentation.  
- A **Flask web app API** was developed to integrate the entire pipeline, simulating a real-world data science project lifecycle.  

---

## 2. Technologies and Tools  
- **Languages & Libraries**: Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn, Optuna, LightGBM, Shap, Flask  
- **Environments & Tools**: Jupyter Notebook, Git/GitHub, Anaconda, VS Code  
- **Concepts Used**: Machine Learning classification, statistics, modular pipelines, CI/CD principles  

---

## 3. Business Problem and Project Objective  

### 3.1 Business Problem  
A bank manager is concerned about **increasing customer churn** in its credit card service.  
The goal is to **predict the likelihood of customer churn** so that retention strategies can be applied proactively.  

### 3.2 Context – Key Performance Indicators (KPIs)  
- **Customer Acquisition Cost (CAC):** Measures marketing/sales costs per acquired customer.  
- **Customer Lifetime Value (CLV):** Expected revenue from a customer during their tenure.  
- **Churn Rate:** Percentage of customers leaving within a given period.  

The bank aims to **reduce CAC and Churn while maximizing CLV**.  

### 3.3 Project Objectives  
- Identify factors driving customer churn.  
- Build a model to predict customer churn probability.  
- Provide actionable strategies to reduce churn.  

### 3.4 Project Benefits  
- Cost Savings  
- Improved Retention  
- Enhanced Customer Experience  
- Targeted Marketing  
- Revenue Protection  

### 3.5 Conclusion  
Predicting **probability scores (instead of binary outcomes)** is more actionable, enabling the bank to focus retention strategies on high-risk customers and allocate resources effectively.  

---

## 4. Solution Pipeline (CRISP-DM Framework)  
1. Define the problem  
2. Collect data and explore  
3. Train-test split  
4. Perform EDA  
5. Feature engineering, cleaning, and preprocessing  
6. Model training, comparison, feature selection, and tuning  
7. Final evaluation  
8. Interpretation of results  
9. Deployment  

---

## 5. Main Business Insights  
- **High churn rate**: ~16% of customers left.  
- **Churners** tend to have **lower credit limits, balances, transactions, and utilization ratios**.  
- **Utilization ratio** is a strong churn indicator—25% of customers had zero utilization, where most churners fall.  
- **Customer contacts**: 75% contacted the bank ≥2 times; churn rate increases with more contacts, and all customers with **≥6 contacts churned**.  
- **Inactivity**: 90% of customers were inactive 1–3 months in the last year.  
- **Doctorate-level customers** have the highest churn rate.  

---

## 6. Modeling  
Two preprocessing pipelines were created:  
- **Linear Models:** One-hot encoding for categorical, standard scaling for numeric.  
- **Tree-based Models:** Ordinal encoding for ordinal categories, target encoding for others, no scaling.  

**Feature engineering** created attributes like average transaction amount, inactive months ratio, and total spending.  

Models tested with **stratified k-fold cross-validation** using ROC-AUC (better for imbalanced targets).  
- **LightGBM** achieved the best validation score → selected for feature selection, hyperparameter tuning, and evaluation.  
- **Feature selection** via **RFE** → reduced to 25 key features.  
- **Hyperparameter tuning** with **Bayesian Search**.  
- **Class weights** adjusted to handle imbalanced churn class.  

### Final LightGBM Results  
- **Recall:** 0.89 (correctly identifies 89% churners)  
- **Precision:** 0.90 (90% of predicted churners are actual churners)  
- **Accuracy:** 0.9659  
- **ROC-AUC:** 0.9913  
- **Confusion Matrix:** 290/325 churners correctly predicted, 297/324 predicted churners correct  

| Model     | Accuracy | Precision | Recall | F1-Score | ROC-AUC | KS      | Gini    | PR-AUC  | Brier   |  
|-----------|----------|-----------|--------|----------|---------|---------|---------|---------|---------|  
| LightGBM  | 0.965943 | 0.895062  | 0.892308 | 0.893683 | 0.991279 | 0.898897 | 0.982559 | 0.964932 | 0.025852 |  

**Interpretability:** Using **SHAP values**, features confirmed EDA insights—e.g., lower transaction counts strongly increase churn risk.  

---

## 7. Financial Results  
Estimated revenue gain = **$171,477**.  
- Gains: True positives → 10% retention fee on balances.  
- Costs: False positives → 8% discount offered.  
- Losses: False negatives → 18% lost fee revenue.  

Result: Project provides **significant financial value**.  

---

## 8. Web App and Next Steps  
- Built a **Flask API web app** for predicting churn probability given customer inputs.  
- Plans for deployment on **AWS Elastic Beanstalk** (.ebextensions/config.py already defined).  
- Added **logging and monitoring** for production use.  

---

## 9. Run Locally  

### Prerequisites  
- Python 3.11.4  
- pip  
- Git  

### Steps  
```bash
# Clone repository  
git clone https://github.com/allmeidaapedro/Churn-Prediction-Credit-Card.git  

# Navigate into repo  
cd Churn-Prediction-Credit-Card  

# Create virtual environment  
python -m venv venv  

# Activate (Linux/Mac)  
source venv/bin/activate  

# Activate (Windows)  
venv\Scripts\activate  

# Install dependencies  
pip install -r requirements.txt  

# Run application  
python application.py  

# 10. Dataset link
The dataset was collected from kaggle.

Link: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers?sort=votes

