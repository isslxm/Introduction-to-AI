import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import argparse

def generate_data(num_samples=200):
    np.random.seed(42)
    # Признаки: уровень сахара в крови, давление
    X = np.random.rand(num_samples, 2) * 100
    y = ((X[:, 0] > 80) & (X[:, 1] > 85)).astype(int)
    return X, y

def train_model(X, y, model_path='diabetes_model.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Оценка модели
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Сохранение модели
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")
    return clf

def predict_patient(model, sugar_level, pressure):
    sample = np.array([[sugar_level, pressure]])
    prediction = model.predict(sample)
    return "Risk" if prediction[0] else "No risk"

def interactive_mode(model):
    print("\nInteractive mode (enter 'q' to quit)")
    while True:
        try:
            input_str = input("Enter sugar level and blood pressure (separated by space): ")
            if input_str.lower() == 'q':
                break
            
            sugar, pressure = map(float, input_str.split())
            result = predict_patient(model, sugar, pressure)
            print(f"Prediction: {result}")
            
        except ValueError:
            print("Invalid input. Please enter two numbers separated by space.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Diabetes Risk Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train and save the model')
    parser.add_argument('--model', type=str, default='diabetes_model.pkl', 
                       help='Path to model file (default: diabetes_model.pkl)')
    parser.add_argument('--sugar', type=float, help='Sugar level for prediction')
    parser.add_argument('--pressure', type=float, help='Blood pressure for prediction')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.train:
        X, y = generate_data()
        train_model(X, y, args.model)
    
    if os.path.exists(args.model):
        model = joblib.load(args.model)
        print(f"Model loaded from {args.model}")
        
        if args.sugar is not None and args.pressure is not None:
            result = predict_patient(model, args.sugar, args.pressure)
            print(f"\nPrediction for sugar={args.sugar}, pressure={args.pressure}: {result}")
        
        if args.interactive:
            interactive_mode(model)
    elif not args.train:
        print(f"Model file {args.model} not found. Use --train to create a new model.")

if __name__ == "__main__":
    main()