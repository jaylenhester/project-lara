import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, demographic_parity_ratio
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lara - Bias Audit", layout="wide")

st.title("Lara: Machine Learning Bias Audit Tool")
st.markdown("Analyzing fairness in machine learning models")

# Load and train model (cached so it doesn't rerun)

@st.cache_data
def load_and_train():
    # load dataset

    df = pd.read_csv('data/adult_clean.csv')
    X = df.drop(columns=['class'])
    y = (df['class'] == '>50K').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return X_test, y_test, y_pred

X_test, y_test, y_pred = load_and_train()

# sidebar
st.sidebar.header("Select Protected Attribute")
attribute = st.sidebar.selectbox("Audit by:", ["sex", "race"])

# main content

col1,col2 = st.columns(2)

with col1: 
    st.subheader("Model Performance")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Overall Accuracy", value=f"{accuracy:.1%}")

with col2: 
    st.subheader("Fairness Check")
    sensitive = X_test[attribute]
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive)
    st.metric("Demographic Parity Ratio", value=f"{dp_ratio:.3f}")
    if dp_ratio < 0.8:
        st.error("FAILS 80% rule")
    else:
        st.success("PASSES 80% rule")

# detailed breakdown
st.subheader(f"Selection Rate by {attribute.title()}")

mf = MetricFrame(
    metrics={'selection_rate': selection_rate, 'true_positive_rate': true_positive_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive
)

st.dataframe(mf.by_group.style.format("{:.3f}"))

# chart
fig, ax = plt.subplots(figsize=(10,4))
data = mf.by_group['selection_rate'].sort_values()

x_vals = data.values.tolist()
y_vals = data.index.tolist()
threshold = max(x_vals) * 0.8

bars = ax.barh(y_vals, x_vals, color='steelblue')
ax.axvline(x=threshold, color='red', linestyle='--', label='80% Threshold')
ax.set_xlabel('Selection Rate')
ax.set_title(f'Selection Rate by {attribute.title()}')
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("*Lara v1.0 - Built by Jaylen Hester*")