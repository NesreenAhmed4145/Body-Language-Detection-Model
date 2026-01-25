import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# 1. Load Data
print("⏳ Loading data...")
df = pd.read_csv('body_language_dataset_imges.csv')

print(f"✅ Loaded {len(df)} rows of data.")

# 2. Prepare Data
X = df.drop('label', axis=1) # Features (Coordinates)
y = df['label'] # Target (Class Names)

# Split Data: 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Setup Training Pipelines
# We create a dictionary to hold both models for comparison
pipelines = {
    'RandomForest': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True)) 
    # probability=True is important for confidence scores later
}

# 4. Train and Compare Models
print("\n🚀 Training and comparing models...")
fit_models = {}
best_model = None
best_accuracy = 0.0
best_model_name = ""

for algo, pipeline in pipelines.items():
    # Train the model
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"➡️  {algo} Accuracy: {accuracy*100:.2f}%")
    
    # Check if this model is the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = algo

# 5. Final Result and Saving
print(f"\n🏆 The winner is: {best_model_name} with accuracy {best_accuracy*100:.2f}%")

print("💾 Saving the best model...")
with open('body_language_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ {best_model_name} model saved successfully as body_language_model.pkl")