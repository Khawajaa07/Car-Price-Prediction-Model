import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data
@st.cache_data

def load_data():
    file_path = Path(__file__).parent / "toyota.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Title
st.title("🚗 Toyota Car Price Prediction Dashboard")

# Preview Data
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Dataset Info
st.subheader("Dataset Info")
st.write("Shape:", df.shape)
st.write("Data Types:")
st.write(df.dtypes)

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Summary Stats
st.subheader("Summary Statistics")
st.write(df.describe())

# EDA: Histograms
st.subheader("Histograms")
numeric_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
st.pyplot(fig)

# Box plots
st.subheader("Box Plots")
fig, ax = plt.subplots(figsize=(10, 6))
df[numeric_cols].boxplot(ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Preprocessing
df_model = df.copy()
categorical_cols = ['model', 'transmission', 'fuelType']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

# Features and target
X = df_model.drop('price', axis=1)
y = df_model['price']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
st.subheader("Model Performance")
st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# Feature Importance
st.subheader("Feature Importance")
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_imp, x='Importance', y='Feature', ax=ax)
st.pyplot(fig)

# Interactive prediction
st.subheader("Predict Car Price")
with st.form("prediction_form"):
    year = st.slider("Year", int(df['year'].min()), int(df['year'].max()), 2017)
    mileage = st.slider("Mileage", int(df['mileage'].min()), int(df['mileage'].max()), 30000)
    tax = st.slider("Tax", 0, 600, 150)
    mpg = st.slider("MPG", 10.0, 100.0, 50.0)
    engine_size = st.slider("Engine Size", 0.0, 5.0, 1.5)
    model_val = st.selectbox("Model", le_dict['model'].classes_)
    trans_val = st.selectbox("Transmission", le_dict['transmission'].classes_)
    fuel_val = st.selectbox("Fuel Type", le_dict['fuelType'].classes_)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame({
        'model': [le_dict['model'].transform([model_val])[0]],
        'year': [year],
        'transmission': [le_dict['transmission'].transform([trans_val])[0]],
        'mileage': [mileage],
        'fuelType': [le_dict['fuelType'].transform([fuel_val])[0]],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engine_size],
    })
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Car Price: £{prediction:,.2f}")

# Conclusion
st.subheader("Insights")
st.markdown("""
- Car price is most influenced by year, mileage, and engine size.
- Fuel type and transmission also have moderate influence.
- This dashboard helps users estimate car resale value interactively.
""")
