import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

# Load dataset
data = pd.read_csv('winequality-red.csv')  # update path if necessary

# Separate features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and SVM
pipe = make_pipeline(StandardScaler(), SVC())

# Define hyperparameters for grid search
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_micro')
grid_search.fit(X_train, y_train)

# Best parameters and scores
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1 Score:", grid_search.best_score_)

# Evaluate on test data
y_pred = grid_search.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Score:", f1_score(y_test, y_pred, average='micro'))
