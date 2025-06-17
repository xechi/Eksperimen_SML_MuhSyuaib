import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Gantilah nilai nol dengan NaN untuk kolom medis yang tidak mungkin bernilai nol
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    # Imputasi missing value dengan median
    df.fillna(df.median(), inplace=True)

    # Pisahkan fitur dan target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['Outcome'] = y.reset_index(drop=True)

    # Simpan hasil ke file CSV
    output_path = "preprocessing/diabetes_clean.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)

    print(f"Preprocessed data successfully saved to: {output_path}")
    return df_scaled

if __name__ == "__main__":
    preprocess_data("diabetes.csv")