import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("../data/zomato-cleaned.csv", encoding='latin1')

# Create target
dataset['Successful'] = dataset['Aggregate rating'].apply(lambda x: 1 if x > 4.0 else 0)

# Binary encoding
dataset['Has Table booking'] = dataset['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
dataset['Has Online delivery'] = dataset['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)

# Fill missing values
dataset['Cuisines'] = dataset['Cuisines'].fillna('Unknown')
dataset['Currency'] = dataset['Currency'].fillna('INR')
dataset['City'] = dataset['City'].fillna('Other')
dataset['Locality'] = dataset['Locality'].fillna('Unknown')

# Label Encoding
label_encoders = {
    'City': LabelEncoder(),
    'Cuisines': LabelEncoder(),
    'Locality': LabelEncoder()
}

dataset['City'] = label_encoders['City'].fit_transform(dataset['City'].astype(str))
dataset['Cuisines'] = label_encoders['Cuisines'].fit_transform(dataset['Cuisines'].astype(str))
dataset['Locality'] = label_encoders['Locality'].fit_transform(dataset['Locality'].astype(str))
dataset['Currency'] = label_encoders['City'].fit_transform(dataset['Currency'].astype(str))  # Reusing encoder

# Features and target
X = dataset.drop(['Aggregate rating', 'Successful'], axis=1)
y = dataset['Successful']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Initial model for GridSearch
model_for_search = RandomForestClassifier(class_weight='balanced', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model_for_search, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_sm, y_train_sm)
print("\n✅ Best Parameters from Grid Search:", grid_search.best_params_)

# Best model before feature selection
best_model_raw = grid_search.best_estimator_

# Feature Selection
selector = SelectFromModel(best_model_raw, prefit=True, threshold='median')
X_train_reduced = selector.transform(X_train_sm)
X_test_reduced = selector.transform(X_test)

# Print selected feature names
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
print("\n✅ Selected Features after Feature Selection:")
print(selected_features.tolist())

# Retrain on reduced features
best_model_raw.fit(X_train_reduced, y_train_sm)

# Predict probabilities and evaluate
y_probs = best_model_raw.predict_proba(X_test_reduced)[:, 1]
custom_threshold = 0.35
y_pred_custom = (y_probs >= custom_threshold).astype(int)

# Final evaluation
print(f"\n✅ Evaluation with threshold = {custom_threshold}")
print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))

# Save final model and metadata
with open("../data/final_model.sav", "wb") as f:
    pickle.dump(best_model_raw, f)

with open("../data/label_encoders.sav", "wb") as f:
    pickle.dump(label_encoders, f)

with open("../data/selector.sav", "wb") as f:
    pickle.dump(selector, f)

with open("../data/threshold.txt", "w") as f:
    f.write(str(custom_threshold))

