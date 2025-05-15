import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath):
    # Cargar dataset
    df = pd.read_csv(filepath)
    
    # Eliminar filas con valores faltantes en Diabetes
    df = df.dropna(subset=['Diabetes'])
    
    # Convertir Diabetes a entero (0 o 1)
    df['Diabetes'] = df['Diabetes'].astype(int)
    
    # Filtrar valores extremos en BMI
    df = df[(df['BMI'] >= 10) & (df['BMI'] <= 50)]
    
    # Variables numéricas para normalizar
    num_features = ['BMI', 'PhysHlth', 'MentHlth', 'Age']
    
    # Normalización
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    
    # Codificación variables categóricas
    cat_features = ['Smoker', 'Sex']
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

if __name__ == "__main__":
    filepath = '../data/diabetes_data.csv'
    df_clean = load_and_clean_data(filepath)
    print(df_clean.head())
