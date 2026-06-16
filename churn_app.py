import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

warnings.filterwarnings('ignore')

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #e67e22; font-weight: bold; }
    .risk-low    { color: #27ae60; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
@st.cache_data
def load_and_train(df_raw):
    df = df_raw.copy()
    df = df.drop(columns=['customerID'], errors='ignore')

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_all_sc   = scaler.transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_sc, y_train)

    return model, scaler, X, y, X_test_sc, y_test, X_all_sc, cat_cols


def assign_risk(prob):
    if prob >= 0.60:
        return 'High Risk'
    elif prob >= 0.30:
        return 'Medium Risk'
    else:
        return 'Low Risk'


def risk_color(risk):
    return {'High Risk': '#e74c3c', 'Medium Risk': '#e67e22', 'Low Risk': '#27ae60'}[risk]


# --------------------------------------------------
# Sidebar — Dataset Upload
# --------------------------------------------------
st.sidebar.title("Telco Churn Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("### Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload telco_churn_features.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.sidebar.info("Upload your CSV file to get started.")
    st.title("Telco Customer Churn Dashboard")
    st.markdown("Upload `telco_churn_features.csv` from the sidebar to load the dashboard.")
    st.stop()

# --------------------------------------------------
# Load Data & Train
# --------------------------------------------------
df_raw = pd.read_csv(uploaded_file)

# Normalize Churn to 0/1
if df_raw['Churn'].dtype == object:
    df_raw['Churn'] = df_raw['Churn'].map({'Yes': 1, 'No': 0})

with st.spinner("Training model..."):
    model, scaler, X, y, X_test_sc, y_test, X_all_sc, cat_cols = load_and_train(df_raw)

# Predictions on full dataset
churn_proba = model.predict_proba(X_all_sc)[:, 1]

pred_df = pd.DataFrame({
    'CustomerID'      : df_raw['customerID'] if 'customerID' in df_raw.columns else range(len(df_raw)),
    'MonthlyCharges'  : df_raw['MonthlyCharges'].values,
    'ActualChurn'     : y.values,
    'ChurnProbability': np.round(churn_proba, 4)
})
pred_df['RiskCategory'] = pred_df['ChurnProbability'].apply(assign_risk)

# --------------------------------------------------
# Sidebar — Navigation
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Risk Categories", "Revenue Impact", "Model Performance", "Customer Lookup"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Customers:** {len(df_raw):,}")
st.sidebar.markdown(f"**Churn Rate:** {y.mean():.1%}")
st.sidebar.markdown(f"**Model:** Logistic Regression")
st.sidebar.markdown(f"**ROC-AUC:** 0.84")


# ==================================================
# PAGE 1 — Overview
# ==================================================
if page == "Overview":
    st.title("Telco Customer Churn — Dashboard")
    st.markdown("Internship ML Project | Day 5")
    st.markdown("---")

    high   = (pred_df['RiskCategory'] == 'High Risk').sum()
    medium = (pred_df['RiskCategory'] == 'Medium Risk').sum()
    low    = (pred_df['RiskCategory'] == 'Low Risk').sum()
    monthly_at_risk = pred_df[pred_df['RiskCategory'] == 'High Risk']['MonthlyCharges'].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",       f"{len(pred_df):,}")
    c2.metric("Churn Rate",            f"{y.mean():.1%}")
    c3.metric("High Risk",             f"{high:,}",    delta=f"{high/len(pred_df):.1%}", delta_color="inverse")
    c4.metric("Medium Risk",           f"{medium:,}",  delta=f"{medium/len(pred_df):.1%}", delta_color="off")
    c5.metric("Monthly Revenue at Risk", f"${monthly_at_risk:,.0f}", delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Churn Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        churn_counts = pd.Series(y).map({0: 'No Churn', 1: 'Churn'}).value_counts()
        ax.bar(churn_counts.index, churn_counts.values,
               color=['#27ae60', '#e74c3c'], edgecolor='white', width=0.4)
        for i, (label, val) in enumerate(churn_counts.items()):
            ax.text(i, val + 30, str(val), ha='center', fontweight='bold')
        ax.set_ylabel('Customers')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Actual Churn Distribution', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Churn Probability Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(pred_df['ChurnProbability'], bins=40, color='#3498db', edgecolor='white', alpha=0.85)
        ax.axvline(0.30, color='#e67e22', linestyle='--', linewidth=1.5, label='0.30 threshold')
        ax.axvline(0.60, color='#e74c3c', linestyle='--', linewidth=1.5, label='0.60 threshold')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Customers')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Predicted Churn Probability', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================================================
# PAGE 2 — Risk Categories
# ==================================================
elif page == "Risk Categories":
    st.title("Risk Categories")
    st.markdown("---")

    risk_summary = pred_df['RiskCategory'].value_counts().reindex(['High Risk', 'Medium Risk', 'Low Risk'])

    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk",   f"{risk_summary['High Risk']:,}",   f"{risk_summary['High Risk']/len(pred_df):.1%}")
    c2.metric("Medium Risk", f"{risk_summary['Medium Risk']:,}", f"{risk_summary['Medium Risk']/len(pred_df):.1%}")
    c3.metric("Low Risk",    f"{risk_summary['Low Risk']:,}",    f"{risk_summary['Low Risk']/len(pred_df):.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#e74c3c', '#e67e22', '#27ae60']
        bars = ax.bar(risk_summary.index, risk_summary.values, color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, risk_summary.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(val), ha='center', fontweight='bold')
        ax.set_title('Customers by Risk Category', fontweight='bold')
        ax.set_ylabel('Number of Customers')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(risk_summary.values, labels=risk_summary.index,
               autopct='%1.1f%%', colors=colors,
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Risk Category Breakdown', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Risk Category Table (Sample — 20 rows)**")

    filter_risk = st.selectbox("Filter by Risk", ["All", "High Risk", "Medium Risk", "Low Risk"])
    display_df  = pred_df if filter_risk == "All" else pred_df[pred_df['RiskCategory'] == filter_risk]
    st.dataframe(display_df.head(20).reset_index(drop=True), use_container_width=True)

    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Results", csv, "churn_predictions.csv", "text/csv")


# ==================================================
# PAGE 3 — Revenue Impact
# ==================================================
elif page == "Revenue Impact":
    st.title("Revenue Impact Analysis")
    st.markdown("---")

    revenue_by_risk = pred_df.groupby('RiskCategory')['MonthlyCharges'].agg(['sum', 'mean', 'count'])
    revenue_by_risk.columns = ['Monthly Revenue', 'Avg Charges', 'Customers']
    revenue_by_risk = revenue_by_risk.reindex(['High Risk', 'Medium Risk', 'Low Risk'])

    high_monthly = revenue_by_risk.loc['High Risk', 'Monthly Revenue']
    high_annual  = high_monthly * 12

    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk Monthly Revenue", f"${high_monthly:,.0f}")
    c2.metric("High Risk Annual Revenue",  f"${high_annual:,.0f}")
    c3.metric("Avg Charge — High Risk",    f"${revenue_by_risk.loc['High Risk','Avg Charges']:.2f}/mo")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#e74c3c', '#e67e22', '#27ae60']
        bars = ax.bar(revenue_by_risk.index, revenue_by_risk['Monthly Revenue'],
                      color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, revenue_by_risk['Monthly Revenue']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title('Monthly Revenue by Risk Category', fontweight='bold')
        ax.set_ylabel('Monthly Revenue ($)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(revenue_by_risk.index, revenue_by_risk['Avg Charges'],
                      color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, revenue_by_risk['Avg Charges']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'${val:.2f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title('Avg Monthly Charges by Risk Category', fontweight='bold')
        ax.set_ylabel('Avg Monthly Charges ($)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Revenue Summary Table**")
    rev_display = revenue_by_risk.copy()
    rev_display['Monthly Revenue'] = rev_display['Monthly Revenue'].apply(lambda x: f"${x:,.2f}")
    rev_display['Avg Charges']     = rev_display['Avg Charges'].apply(lambda x: f"${x:.2f}")
    st.dataframe(rev_display, use_container_width=True)


# ==================================================
# PAGE 4 — Model Performance
# ==================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Logistic Regression — Selected as best model (Day 3)")
    st.markdown("---")

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.4f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    c3.metric("Recall",    f"{recall_score(y_test, y_pred):.4f}")
    c4.metric("F1 Score",  f"{f1_score(y_test, y_pred):.4f}")
    c5.metric("ROC-AUC",   f"{roc_auc_score(y_test, y_prob):.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    ax=ax, linewidths=0.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)


# ==================================================
# PAGE 5 — Customer Lookup
# ==================================================
elif page == "Customer Lookup":
    st.title("Customer Lookup")
    st.markdown("Search any customer to see their churn probability and risk category.")
    st.markdown("---")

    search_id = st.text_input("Enter Customer ID")

    if search_id:
        match = pred_df[pred_df['CustomerID'].astype(str) == search_id.strip()]
        if len(match) == 0:
            st.warning("Customer ID not found. Please check and try again.")
        else:
            row  = match.iloc[0]
            risk = row['RiskCategory']
            prob = row['ChurnProbability']
            color = risk_color(risk)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customer ID",        str(row['CustomerID']))
            c2.metric("Churn Probability",  f"{prob:.2%}")
            c3.metric("Monthly Charges",    f"${row['MonthlyCharges']:.2f}")
            c4.metric("Risk Category",      risk)

            st.markdown("---")

            # Probability gauge bar
            st.markdown(f"**Churn Risk Level: {risk}**")
            fig, ax = plt.subplots(figsize=(7, 1.2))
            ax.barh([''], [1], color='#ecf0f1', height=0.4)
            ax.barh([''], [prob], color=color, height=0.4)
            ax.axvline(0.30, color='#e67e22', linestyle='--', linewidth=1)
            ax.axvline(0.60, color='#e74c3c', linestyle='--', linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Churn Probability')
            ax.text(prob + 0.01, 0, f'{prob:.0%}', va='center', fontweight='bold', color=color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if risk == 'High Risk':
                st.error("Action Required: This customer is at high risk of churning. Consider an immediate retention offer.")
            elif risk == 'Medium Risk':
                st.warning("Monitor: This customer shows medium churn risk. Proactive engagement recommended.")
            else:
                st.success("Stable: This customer is at low risk. Continue normal engagement.")
    else:
        st.info("Enter a Customer ID above to look up their churn prediction.")
        st.markdown("**Sample Customer IDs from dataset:**")
        st.dataframe(pred_df[['CustomerID', 'ChurnProbability', 'RiskCategory']].head(10), use_container_width=True)
