================================================================
TELCO CUSTOMER CHURN PREDICTION
Internship ML Project
================================================================

OBJECTIVE
---------
Predict which customers are likely to churn and provide
business insights to help reduce customer loss.

----------------------------------------------------------------
FILES
----------------------------------------------------------------

TASK_1_Telco_Churn_v2.ipynb     Main project notebook
churn_app.py                     Streamlit dashboard
telco_churn_features.csv         Dataset (auto-generated)
requirements.txt                 Required packages

----------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------

1. Install dependencies
       pip install -r requirements.txt

2. Run the notebook
       Open TASK_1_Telco_Churn_v2.ipynb and run all cells

3. Run the dashboard
       streamlit run churn_app.py
       Upload telco_churn_features.csv from the sidebar

----------------------------------------------------------------
KEY RESULTS
----------------------------------------------------------------

Dataset        : 7,043 customers | 29 features
Best Model     : Logistic Regression
Recall         : 0.79  |  ROC-AUC : 0.84
High Risk      : 2,350 customers (33.4%)
Revenue at Risk: $182,312/month | $2,187,746/year

----------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------

Python 3.9+
pandas, numpy, scikit-learn, xgboost,
matplotlib, seaborn, shap, streamlit

================================================================
<<<<<<< HEAD
GitHub repository Link 
https://github.com/mfahad-ai/telco-churn-project
=======
>>>>>>> 4cf93c0f3ff6ecbfdff7d931d51d8379d028d9c5
