import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_analysis(filepath):
    # Cargar datos
    df = pd.read_csv(filepath)

    # Mostrar dimensiones
    print(f"\nDataset shape: {df.shape}")
    
    # Estadísticas generales
    print("\nEstadísticas descriptivas (variables numéricas):")
    print(df.describe())

    # Distribución de la variable objetivo
    print("\nDistribución de la variable objetivo 'Diabetes':")
    print(df['Diabetes'].value_counts())

    # Histograma de variables numéricas seleccionadas
    numeric_cols = ['Age', 'BMI', 'PhysHlth', 'MentHlth']
    df[numeric_cols].hist(bins=20, figsize=(10, 6))
    plt.suptitle("Distribuciones de variables numéricas", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Gráfico de barras para 'Diabetes'
    sns.countplot(x='Diabetes', data=df)
    plt.title('Distribución de la variable objetivo (Diabetes)')
    plt.xlabel('Diabetes (0: No, 1: Sí)')
    plt.ylabel('Frecuencia')
    plt.show()

    # Heatmap de correlación
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Matriz de correlación')
    plt.show()

if __name__ == "__main__":
    filepath = os.path.join("..", "data", "diabetes_data.csv")
    run_analysis(filepath)
