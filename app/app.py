import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

# Cargar modelo
model_package = joblib.load("C:/Users/uvire/Desktop/Proyectos IA y DS/fraud-detection-system/src/fraud_model.pkl")
model = model_package['model']
expected_features = model_package["features"]
# Header
st.title("💳 AI Fraud Detection Dashboard")
st.markdown("### Real-time Financial Fraud Detection System")

# Sidebar
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    # Crea una copia solo para el modelo
    model_data = data.copy()

    # Eliminar columna que el modelo no espera
    for col in ['Class','Fraud Prediction','Fraud Probability']:
        if col in model_data.columns:
            model_data = model_data.drop(col, axis=1)

    # Asegurar que las columnas coincidan
    model_data = model_data[expected_features]  
    # Predicción
    predictions = model.predict(model_data)
    probabilities = model.predict_proba(model_data)[:,1]
    #Agregar resultados
    data['Fraud Prediction']= predictions 
    data['Fraud Probability'] = probabilities

    # Métricas
    total = len(data)
    frauds = data['Fraud Prediction'].sum()
    normal = total - frauds

    # 🔥 KPIs tipo dashboard
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected 🚨", frauds)
    col3.metric("Normal Transactions ✅", normal)

    st.divider()

    # 📊 Gráficos
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Fraud vs Normal")
        fig, ax = plt.subplots()
        data['Fraud Prediction'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    with col5:
        st.subheader("Transaction Amount Distribution")
        fig2, ax2 = plt.subplots()
        data['Amount'].hist(bins=50)
        st.pyplot(fig2)

    st.divider()

    # 📋 Tabla
    st.subheader("📊 Data Preview")
    st.dataframe(data.head())

    st.subheader("🚨 Fraudulent Transactions")
    st.dataframe(data[data['Fraud Prediction'] == 1])

else:
    st.info("Please upload a CSV file to start analysis.")