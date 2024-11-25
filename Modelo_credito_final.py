import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import streamlit as st
from multiprocessing import cpu_count

# Cargar datos
file_path = "base_final.xlsx"
data = pd.read_excel(file_path)

##########################################################################################
# Preparar base de datos para el modelo
data = data.dropna(subset=['grade'])

# Definir variables
X = data.drop('grade', axis=1)
y = data['grade']

# Convertir el ingreso anual a formato numérico
X['annual_inc'] = pd.to_numeric(X['annual_inc'], errors='coerce')

# Agrupar las categorías poco comunes en 'title'
frecuencia_titulos = X['title'].value_counts()
titulos_comunes = frecuencia_titulos[frecuencia_titulos > 10].index
X['title'] = X['title'].where(X['title'].isin(titulos_comunes), 'Otros')

# Definir variables categóricas
columnas_categoricas = ['term', 'emp_length', 'title']

# Convertir en variables dummies
X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verificar que no hayan NaNs
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

##########################################################################################
# Configuración para maximizar el uso de RAM y CPU

# Usar todos los hilos disponibles
num_cores = cpu_count()  # Ryzen 9 7940HS: 16 hilos

# Dividir datos en lotes
def split_data(X, y, n_splits):
    split_size = len(X) // n_splits
    splits = []
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i != n_splits - 1 else len(X)
        splits.append((X[start:end], y[start:end]))
    return splits

# Entrenar modelo en un lote
def train_svm(data_subset):
    X_subset, y_subset = data_subset
    # Incrementar cache_size para usar más RAM
    model = svm.SVC(kernel='linear', cache_size=14000)  # 14 GB para cálculos en RAM
    model.fit(X_subset, y_subset)
    return model

# Dividir datos
data_splits = split_data(X_train, y_train, num_cores)

# Entrenar modelos en paralelo
models = Parallel(n_jobs=num_cores)(
    delayed(train_svm)(split) for split in data_splits
)

# Usar el primer modelo como referencia
clf = models[0]

##########################################################################################
# Paralelizar predicciones

# Dividir el conjunto de prueba en lotes
test_batches = np.array_split(X_test, num_cores)

# Predicciones paralelas
def predict_batch(model, X_batch):
    return model.predict(X_batch)

y_pred_batches = Parallel(n_jobs=num_cores)(
    delayed(predict_batch)(clf, batch) for batch in test_batches
)

# Combinar predicciones
y_pred = np.concatenate(y_pred_batches)

##########################################################################################
# Iniciar la interfaz Streamlit
st.title("Credit Score Prediction Model")

# Display descriptive statistics
st.subheader("Descriptive Statistics")
fig, ax = plt.subplots()
ax.hist(data["loan_amnt"], bins=20, color="blue", edgecolor="black")
ax.set_title("Histogram of Loan Amounts")
ax.set_xlabel("Amount")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Display model results
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:\n", classification_report(y_test, y_pred))

##########################################################################################
# Input new data
loan_amnt = st.sidebar.number_input("Loan amount (USD): ")
annual_inc = st.sidebar.number_input("Annual income: ")
int_rate = st.sidebar.number_input("Interest rate (%): ")

meses = ["36 months", "60 months"]
term = st.sidebar.selectbox("Loan term (in months):", meses)

periodo_trabajo = ["<1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
                   "7 years", "8 years", "9 years", "10+ years"]
emp_length = st.sidebar.selectbox("Years in current job:", periodo_trabajo)

proposito = ["Business", "Credit card refinancing", "Debt consolidation", "Car financing", "Major purchase",
             "Home improvement", "Home buying", "Medical expenses", "Vacation", "Other"]
title = st.sidebar.selectbox("Loan purpose:", proposito)

# Create dataframe with input data
data_input = pd.DataFrame({
    "loan_amnt": [loan_amnt],
    'annual_inc': [annual_inc],
    'int_rate': [int_rate],
    'term': [term],
    'emp_length': [emp_length],
    'title': [title]
})

data_input = pd.get_dummies(data_input, columns=['term', 'emp_length', 'title'], drop_first=True)

# Ensure columns match
missing_cols = set(X.columns) - set(data_input.columns)
for col in missing_cols:
    data_input[col] = 0
data_input = data_input[X.columns]

# Scale and impute input data
data_input = imputer.transform(data_input)
data_input = scaler.transform(data_input)

# Prediction
prediccion = clf.predict(data_input)
st.write("The predicted credit score is:", prediccion[0])

# Credit approval prediction
def credit_approval(prediccion, loan_amnt, annual_inc):
    # Threshold ranges by category
    ranges = {
        "A": 0.6,
        "B": 0.45,
        "C": 0.4,
        "D": 0.35,
        "E": 0.3,
        "F": 0.25,
        "G": 0.2
    }

    # Get threshold for predicted category
    threshold = ranges.get(prediccion[0], 0)

    # Avoid division by zero if annual income is 0
    if annual_inc == 0:
        return "Error: Annual income cannot be zero."

    # Calculate loan-to-income ratio
    ratio = loan_amnt / annual_inc

    # Decide approval based on ratio
    if ratio <= threshold:
        return "Credit approved"
    else:
        return "Credit not approved"

# Call credit approval function
observation = credit_approval(prediccion, loan_amnt, annual_inc)
st.write(observation)


# python -m streamlit run "C:\1 A documentos erick\ITESO\8vo sem\Modelos de crédito\Proyectos finales\Proyecto final modelos\Modelo_credito_final.py"