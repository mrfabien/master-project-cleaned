import pandas as pd

def data_preparation_ML(X_train, y_train, name_of_variable, levels):
    '''
    This function prepares the data for the ML model. It reshapes the data, drops the rows with nan values, renames the columns by the name of the variables by adding also the stat of the variable (max, min, mean, std), drops the instantaneous variables, u and v wind components, and uses RandomForests to find the most important features.
    '''
    # reshape the X_training into a 2D array

    if y_train == None:
        print('y is None and is not used in the function')
        y_train = X_train[:, :, :4] 

    X_all_2d = X_train.reshape(X_train.shape[0]*X_train.shape[1],X_train.shape[2])
    # same for y_all_3d
    y_all_2d = y_train.reshape(y_train.shape[0]*y_train.shape[1],y_train.shape[2])

    # drop all the rows with nan values

    X_all_2d = pd.DataFrame(X_all_2d)
    X_all_2d_non_na = X_all_2d.dropna()
    y_all_2d = pd.DataFrame(y_all_2d)
    y_all_2d_non_na = y_all_2d.dropna()

    # rename the columns by the name of the variables by adding also the stat of the variable (max, min, mean, std)

    stats = ['max', 'min', 'mean', 'std']
    var_stat = []
    var_stat_all = []

    for var in name_of_variable['variables']:
        for stat in stats:
            var_stat = f'{var}_{stat}'
            var_stat_all.append(var_stat)

    # drop instantaneous variables, u and v wind components

    var_stat_all_x = [var for var in var_stat_all if 'inst' not in var]
    instantaneous = ['instantaneous_10m_wind_gust_max', 'instantaneous_10m_wind_gust_min', 'instantaneous_10m_wind_gust_mean', 'instantaneous_10m_wind_gust_std']

    X_all_2d_non_na.columns = var_stat_all_x
    y_all_2d_non_na.columns = instantaneous

    y_mean = y_all_2d_non_na.drop(columns=['instantaneous_10m_wind_gust_max', 'instantaneous_10m_wind_gust_min', 'instantaneous_10m_wind_gust_std'])
    y_max = y_all_2d_non_na.drop(columns=['instantaneous_10m_wind_gust_min', 'instantaneous_10m_wind_gust_mean', 'instantaneous_10m_wind_gust_std'])

    # Ensure index alignment
    levels.reset_index(drop=True, inplace=True)

    # Pop or drop the first level if it's 0
    if levels['levels'].iloc[0] == 0:  # Use iloc for positional indexing
        levels = levels.iloc[1:]
        
    # Reset the index to ensure alignment
    levels.reset_index(drop=True, inplace=True)
    levels_below_300 = levels[levels['levels'] <= 300]

    # drop columns with 10m_u_component_of_wind and 10m_v_component_of_wind variables + specific_rain_water_content from level 10 to 300 (no values)

#    X_all_2d_non_na = X_all_2d_non_na.drop(columns=['10m_u_component_of_wind_max', 
#                                                    '10m_u_component_of_wind_min', 
#                                                    '10m_u_component_of_wind_mean', 
#                                                    '10m_u_component_of_wind_std', 
#                                                    '10m_v_component_of_wind_max', 
#                                                    '10m_v_component_of_wind_min', 
#                                                    '10m_v_component_of_wind_mean', 
#                                                    '10m_v_component_of_wind_std',
#    ])
    X_all_2d_non_na = X_all_2d_non_na.drop(columns=[f'specific_rain_water_content_{level}_max' for level in levels_below_300['levels']])
    X_all_2d_non_na = X_all_2d_non_na.drop(columns=[f'specific_rain_water_content_{level}_mean' for level in levels_below_300['levels']])
    X_all_2d_non_na = X_all_2d_non_na.drop(columns=[f'specific_rain_water_content_{level}_min' for level in levels_below_300['levels']])
    X_all_2d_non_na = X_all_2d_non_na.drop(columns=[f'specific_rain_water_content_{level}_std' for level in levels_below_300['levels']])

    var_stat_all_x = [
        var for var in var_stat_all_x 
        if all(
            f'specific_rain_water_content_{level}_' not in var 
            for level in levels_below_300['levels']
        )
    ]
    # using RandomForests to find the most important features

    X_standardized = (X_all_2d_non_na - X_all_2d_non_na.mean()) / X_all_2d_non_na.std()
    y_standardized_max = (y_max - y_max.mean()) / y_max.std()
    y_standardized_mean = (y_mean - y_mean.mean()) / y_mean.std()

    return X_standardized, y_standardized_max, y_standardized_mean, var_stat_all_x