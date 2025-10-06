import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os
import json

# --- Configuration ---
PROCESSED_DATA_FOLDER = 'data/processed'
PARAMS_FILE = 'best_params.json'
RANDOM_STATE = 42

def tune_hyperparameters():
    """
    Loads training data and uses GridSearchCV to find the best hyperparameters.
    """
    print("Starting hyperparameter tuning...")

    # Load the processed training data
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv')).values.ravel()

    # Define the parameter grid to search
    # This is a small grid for demonstration purposes. You can expand it.
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the model and GridSearchCV
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='r2')

    # Run the grid search
    print("Running GridSearchCV... (This can take a significant amount of time)")
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"✅ Best parameters found: {best_params}")

    # Save the best parameters to a file
    with open(PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to '{PARAMS_FILE}'")

if __name__ == '__main__':
    tune_hyperparameters()