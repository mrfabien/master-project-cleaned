from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
import os
import sys
custom_library_path = os.path.abspath('util/feature_selection/')
sys.path.append(custom_library_path)

import selection_vars

def process_xgboost_workflow(X_train_pca, X_validation_pca, y_train, y_validation, variable_counts, target_types, param_grid, print_info='yes'):
    results = {}
    for target_type in target_types:  # e.g., ['cdf', 'max']
        for var_count in variable_counts:  # e.g., [20, 30, 40]
            # Convert PCA data to numpy
            X_train_np = X_train_pca[var_count].to_numpy()
            X_validation_np = X_validation_pca[var_count].to_numpy()
            
            # Initialize and train model
            model = XGBRegressor(random_state=42, n_jobs=-1)
            model.fit(X_train_np, y_train[target_type])
            
            # Predictions
            predictions = model.predict(X_validation_np)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_validation[target_type], predictions))
            mae = mean_absolute_error(y_validation[target_type], predictions)
            
            # Store results
            results[f'{target_type}_{var_count}'] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
            }
            
            # Hyperparameter tuning
            search = HalvingGridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)
            search.fit(X_train_np, y_train[target_type])
            results[f'{target_type}_{var_count}']['best_params'] = search.best_params_
            results[f'{target_type}_{var_count}']['search'] = search
            
            # Feature Selection
            selected_vars = selection_vars.feature_selection(
                X_train_pca[var_count],
                X_train_np,
                y_train[target_type],
                model
            )
            results[f'{target_type}_{var_count}']['selected_features'] = selected_vars
            
            if print_info=='yes':
                # Log results
                print(f"Target: {target_type}, Variables: {var_count}")
                print(f"RMSE: {rmse}, MAE: {mae}")
                print(f"Selected Features: {selected_vars}")
                print('-' * 30)
    
    return results