import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    # Load Data
    data_path = "dataset/transportation_data_processed.csv"
    if not os.path.exists(data_path):
        print("Error: File data tidak ditemukan.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['Accident_Occurred'])
    y = df['Accident_Occurred']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hapus folder output lama jika ada
    model_output_path = "model_output"
    if os.path.exists(model_output_path):
        shutil.rmtree(model_output_path)

    # Training
    print("Training model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Simpan Model ke Folder Lokal
    mlflow.sklearn.save_model(model, model_output_path)
    
    print(f"Model berhasil disimpan di folder: {model_output_path}")

if __name__ == "__main__":
    train()