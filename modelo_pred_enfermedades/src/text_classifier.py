import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack

# 1. Cargar datos
df = pd.read_csv("../data/diabetes_data_with_notes.csv")

# 2. Separar características
text = df["Clinical_Notes"]
y = df["Diabetes"]

# Seleccionamos columnas numéricas relevantes (puedes ajustar esta parte)
numerical_cols = df.drop(columns=["Diabetes", "Clinical_Notes"]).select_dtypes(include="number")
X_num = numerical_cols.values

# 3. Vectorizar texto (TF-IDF)
vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(text)

# 4. Escalar los datos numéricos
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# 5. Combinar texto y datos numéricos
X_combined = hstack([X_text, X_num_scaled])

# 6. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluar
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Curva ROC ---
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --- Gráfico de barras: distribución de predicciones ---
pd.Series(y_pred).value_counts().sort_index().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xticks([0, 1], ['No Diabético', 'Diabético'], rotation=0)
plt.ylabel('Cantidad de Predicciones')
plt.title('Distribución de Predicciones del Modelo')
plt.tight_layout()
plt.show()