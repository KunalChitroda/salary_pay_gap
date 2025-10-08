# src/hyperparameter_tuning.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import os
import sys
import yaml
import json

def tune_hyperparameters(params_path):
    """
    Loads training data and uses RandomizedSearchCV to find the best hyperparameters.
    """
    print("Starting hyperparameter tuning...")

    # Load parameters from the YAML file
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # --- Configuration from params.yaml ---
    PROCESSED_DATA_FOLDER = 'data/processed'
    PARAMS_FILE = 'best_params.json'
    RANDOM_STATE = params['preprocess']['random_state']
    PARAM_GRID = params['tune']['param_grid']
    N_ITER = params['tune']['n_iter']
    CV = params['tune']['cv']

    # Load the processed training data
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv')).values.ravel()

    # Initialize the model and RandomizedSearchCV
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=PARAM_GRID,
                                       n_iter=N_ITER, cv=CV, verbose=2,
                                       random_state=RANDOM_STATE, n_jobs=-1, scoring='r2')

    # Run the randomized search
    print(f"Running RandomizedSearchCV for {N_ITER} iterations...")
    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_
    print(f"✅ Best parameters found: {best_params}")

    # Save the best parameters to a file
    with open(PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to '{PARAMS_FILE}'")

if __name__ == '__main__':
    tune_hyperparameters(params_path=sys.argv[1])   