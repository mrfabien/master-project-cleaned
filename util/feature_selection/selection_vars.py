import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector


#for name_of_variable_levels in [vars_selected_40, vars_selected_30, vars_selected_20]:
def extract_timeseries(name_of_variable_levels, all_series='yes'): 
    # shape all_data as the timeseries data
    X_all_vars = {}

    for var in name_of_variable_levels:
        var, stat = var.rsplit("_", 1)
        if var == 'sea_surface_temperature':
            continue
        #for stat in stats:
        var_stat = f'{var}_{stat}'
        #if var_stat == f'instantaneous_10m_wind_gust_{stat}':
        #storm_data = []
        if all_series == 'yes':
            var_temp_non_eu = pd.read_csv(f'data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')
            # reverse the order of the columns
            var_temp_non_eu = var_temp_non_eu.iloc[:, ::-1]
            var_temp_eu = pd.read_csv(f'data/time_series_1h_EU/{var}/{var}_{stat}.csv')
            var_temp = pd.concat([var_temp_non_eu, var_temp_eu], axis=1)
        else:
            var_temp = pd.read_csv(f'data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')
        var_temp = var_temp.drop(columns=['Unnamed: 0', 'storm_index'])
        # select only the 12 first hours (represented by the 12 first columns)
        if all_series == 'no':
            var_temp = var_temp.iloc[:, :12]
        #var_temp_2 = var_temp.drop(columns=['storm_index'])
        var_temp_reshape = var_temp.to_numpy().reshape((var_temp.shape[0]*var_temp.shape[1]))
        X_all_vars[f'{var_stat}'] = var_temp_reshape

    # Assuming all_vars is populated as a dictionary
    df_X_all_vars = pd.DataFrame(X_all_vars)

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(df_X_all_vars)
    return df_X_all_vars, scaled_X

def extract_timeseries_y(all_series='yes'):
    y_all = {}

    var = 'instantaneous_10m_wind_gust'
    stat = 'max'
    var_stat = f'{var}_{stat}'
    if all_series == 'yes':
        var_temp_non_eu = pd.read_csv(f'data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')
        # reverse the order of the columns
        var_temp_non_eu = var_temp_non_eu.iloc[:, ::-1]
        var_temp_eu = pd.read_csv(f'data/time_series_1h_EU/{var}/{var}_{stat}.csv')
        var_temp = pd.concat([var_temp_non_eu, var_temp_eu], axis=1)
    else:
        var_temp = pd.read_csv(f'data/time_series_1h_EU/{var}/{var}_{stat}.csv')
    var_temp = var_temp.drop(columns=['Unnamed: 0', 'storm_index', 'storm_name','start_date'])
    # select only the 12 first hours (represented by the 12 first columns)
    if all_series == 'no':
        var_temp = var_temp.iloc[:, :12]
    #var_temp_2 = var_temp.drop(columns=['storm_index'])
    var_temp_reshape = var_temp.to_numpy().reshape((var_temp.shape[0]*var_temp.shape[1]))
    y_all[f'{var_stat}'] = var_temp_reshape

    df_y = pd.DataFrame(y_all)

    return df_y

def feature_selection(df_X_all_vars, scaled_X, df_y, model):
    # Initialize the sequential feature selector
    sfs = SequentialFeatureSelector(
                                    model,  # Use the same model as before
                                    n_features_to_select=5,  # Select 5 features
                                    direction='forward',  # Forward selection
                                    n_jobs=-1  # Use all available cores
                                    )   

    # Fit the selector

    try: 
        df_y = df_y.to_numpy().ravel()
    except:
        print('y is already a numpy array')
    sfs.fit(scaled_X, df_y)

    # Get the selected feature indices
    selected_indices = sfs.get_support(indices=True)

    # Get the names of the selected features
    selected_features = df_X_all_vars.columns[selected_indices]
    print("Selected features:", selected_features)
    return selected_features