import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    mlflow.autolog()

    # Load Data
    data_path = "transportation_data_processed.csv" 
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan.")
        return

    df = pd.read_csv(data_path)
    
    # Pisahkan Fitur dan Target
    X = df.drop(columns=['Accident_Occurred'])
    y = df['Accident_Occurred']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow Run
    with mlflow.start_run():
        print("Training model...")
        
        # Inisialisasi Model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Training Model
        model.fit(X_train, y_train)

        print("Training selesai")

if __name__ == "__main__":
    train()