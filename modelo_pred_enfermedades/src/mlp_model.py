import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

filepath = '../data/diabetes_data.csv'  # O la ruta correcta que mencionaste

print("Ruta usada para cargar datos:", filepath)
print("Ruta absoluta:", os.path.abspath(filepath))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    print("Buscando archivo en:", os.path.abspath(filepath))
    df = pd.read_csv(filepath)

    # Separar características y variable objetivo
    X = df.drop('Diabetes', axis=1)
    y = df['Diabetes']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_mlp(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # salida binaria

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    filepath = '../data/diabetes_data.csv'  
    X_train, X_test, y_train, y_test = load_data(filepath)

    model = build_mlp(X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int)

    print("\nReporte clasificacion:")
    print(classification_report(y_test, y_pred_labels))

    print("🔍 ROC AUC:", roc_auc_score(y_test, y_pred))

    # Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo MLP')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Gráfico de barras: distribución de predicciones
pd.Series(y_pred_labels.flatten()).value_counts().sort_index().plot(
    kind='bar', color=['skyblue', 'salmon']
)
plt.xticks([0, 1], ['No Diabético', 'Diabético'], rotation=0)
plt.ylabel('Cantidad de Predicciones')
plt.title('Distribución de Predicciones - Modelo MLP')
plt.tight_layout()
plt.show()