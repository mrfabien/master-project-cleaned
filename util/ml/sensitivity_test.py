from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import HalvingGridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
custom_library_path = os.path.abspath('util/feature_selection/')
sys.path.append(custom_library_path)

import selection_vars

def process_xgboost_workflow_old(X_train_pca, X_validation_pca, y_train, y_validation, variable_counts, target_types, param_grid, print_info='yes'):
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
 
def process_xgboost_workflow(X_train_pca, X_validation_pca, y_train, y_validation, variable_counts, target_types, param_grid, print_info='yes'):
    results = {}
    for target_type in target_types:  # e.g., ['cdf', 'max']
        for var_count in variable_counts:  # e.g., [20, 30, 40]
            # Convert PCA data to numpy
            X_train_np = X_train_pca[var_count].to_numpy()
            X_validation_np = X_validation_pca[var_count].to_numpy()
            
            # Initialize and train model
            model = XGBRegressor(n_jobs=-1, random_state=42)
            model.fit(X_train_np, y_train[target_type])
            
            # Predictions
            predictions = model.predict(X_validation_np)
            
            # Metrics before tuning
            rmse = np.sqrt(mean_squared_error(y_validation[target_type], predictions))
            mae = mean_absolute_error(y_validation[target_type], predictions)
            r2 = r2_score(y_validation[target_type], predictions) #model.score(X_validation_np, y_validation[target_type])
            relative_variance_valid = np.var(y_validation[target_type])/(np.mean(y_validation[target_type])**2)
            relative_variance_train = np.var(y_train[target_type])/(np.mean(y_train[target_type])**2)
            
            # Store initial results
            results[f'{target_type}_{var_count}'] = {
                'model': model,
                'rmse_before_tuning': rmse,
                'mae_before_tuning': mae,
                'r2_before_tuning': r2,
                'relative_variance_valid': relative_variance_valid,
                'relative_variance_train': relative_variance_train,
            }
            
            # Hyperparameter tuning
            search = HalvingGridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)
            search.fit(X_train_np, y_train[target_type])
            
            # Get best model after tuning
            best_model = search.best_estimator_
            
            # Predictions after tuning
            tuned_predictions = best_model.predict(X_validation_np)
            
            # Metrics after tuning
            rmse_tuned = np.sqrt(mean_squared_error(y_validation[target_type], tuned_predictions))
            mae_tuned = mean_absolute_error(y_validation[target_type], tuned_predictions)
            r2_tuned = r2_score(y_validation[target_type], tuned_predictions) #best_model.score(X_validation_np, y_validation[target_type])
            
            # Update results
            results[f'{target_type}_{var_count}'].update({
                'best_params': search.best_params_,
                'rmse_after_tuning': rmse_tuned,
                'mae_after_tuning': mae_tuned,
                'r2_after_tuning': r2_tuned,
                'search': search,
            })
            
            # Feature Selection
            selected_vars = selection_vars.feature_selection(
                X_train_pca[var_count],
                X_train_np,
                y_train[target_type],
                best_model
            )
            results[f'{target_type}_{var_count}']['selected_features'] = selected_vars
            
            if print_info == 'yes':
                # Log results
                print(f"Target: {target_type}, Variables: {var_count}")
                print(f"RMSE Before Tuning: {rmse}, MAE Before Tuning: {mae}")
                print(f"RMSE After Tuning: {rmse_tuned}, MAE After Tuning: {mae_tuned}")
                print(f"R2 Before Tuning: {r2}, R2 After Tuning: {r2_tuned}")
                print(f"Relative Variance on validation set: {relative_variance_valid}")
                print(f"Relative Variance on training set: {relative_variance_train}")
                print(f"Best Params: {search.best_params_}")
                print(f"Selected Features: {selected_vars}")
                print('-' * 30)

            #plt.figure(figsize=(10, 6))
            #plt.scatter(y_validation[target_type],tuned_predictions)# , label='Actual')
            ##plt.plot(tuned_predictions, label='Predicted')
            #plt.ylabel('Predicted')
            #plt.xlim(0, 9)
            #plt.xlabel('Actual')
            #plt.ylim(0, 9)
            #plt.title(f'{target_type} - {var_count}')
            #plt.legend()
            #plt.show()
    
    return results