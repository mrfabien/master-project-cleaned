import numpy as np
import pandas as pd
import os
import csv

# Function to read variable names from a CSV file
def read_variable_names(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        variables = [row['variables'] for row in reader]
    return variables

# Function to read values from a specific column in a CSV file
def read_column_values(csv_file, column_name):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        values = [row[column_name] for row in reader]
    return values

# Function to filter rows from one CSV based on values from another CSV
def filter_rows(input_csv, output_csv, column_name, filter_values):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader()  # Write the header to the output file
        for row in reader:
            if row[column_name] in filter_values:
                writer.writerow(row)

def filtering_EU_storms(variables_csv, timestep, path, choosen_directory, levels):

    levels = levels['levels'].to_list()

    variables = read_variable_names(variables_csv)
    # List of storms and levels
    storms = [f"{i}" for i in range(1, 97)]

    # List of statistic types
    stats = ["max", "mean", "min", "std"]

    # Base directories for input CSV files
    base_dir_csv1 = f"{path}data/datasets_{timestep}"
    #if operating_system == 'mac':
        #base_dir_csv2 = f"{path}pre_processing/tracks/ALL_TRACKS/tracks_{timestep}_EU"
        #print(base_dir_csv2)
    #else:
    base_dir_csv2 = f"{path}pre_processing/tracks/ALL_TRACKS/tracks_{timestep}_EU"
    print(base_dir_csv2)

    output_base_dir = f"{path}{choosen_directory}/datasets_{timestep}_EU"

    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Loop through variables, storms, levels, and stats
    for variable in variables:
        for storm in storms:
            for level in levels:
                for stat in stats:
                    csv_file1 = os.path.join(base_dir_csv1, variable, f"storm_{storm}", f"{stat}_{storm}_{level}.csv")
                    if timestep == '1h':
                        csv_file2 = os.path.join(base_dir_csv2, f"storm_{storm}.csv")
                    else:
                        csv_file2 = os.path.join(base_dir_csv2, f"storm_{storm}.csv")
                    output_file = os.path.join(output_base_dir, variable, f"storm_{storm}", f"{stat}_{storm}_{level}.csv")

                    # Create directories for the output file if they do not exist
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    # Only read and filter if the CSV files exist
                    if os.path.exists(csv_file1) and os.path.exists(csv_file2):
                        # Read the column values from the first CSV
                        filter_values = read_column_values(csv_file2, 'timestep')

                        # Filter rows from the second CSV and write to the output CSV
                        filter_rows(csv_file1, output_file, '', filter_values)

                        print(f"Filtered rows for {variable}, {storm}, level {level}, stat {stat} have been written to {output_file}")
                    #else:
                        #print(f"Skipped {variable}, {storm}, level {level}, stat {stat} due to missing files")


def X_y_datasets_EU(name_of_variables, max_time_steps, path, dataset):
    if dataset == 'datasets_1h_EU':
        max_time_steps = 185
    elif dataset == 'datasets_3h_EU':
        max_time_steps = 62
    #max_time_steps = 185 # max_time_steps[col_max].max()+1
    else :
        print('wrong typo')
        return

    def split_variable_level(variable_with_level):
        parts = variable_with_level.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], parts[1]
        else:
            return variable_with_level, 0

    # Initialize the list to hold the data for each storm
    data_list = []
    data_list_y = []

    stats = ['max', 'min', 'mean', 'std']

    for storm_idx in range(1,97):
        storm_data = []
        storm_data_y = []
        for var_name, row in name_of_variables.iterrows():
            var_name = split_variable_level(row['variables'])[0]
            level = split_variable_level(row['variables'])[1]
            if var_name == 'instantaneous_10m_wind_gust':
                try:
                    for stat in stats:
                        df = pd.read_csv(f'{path}/data/{dataset}/{var_name}/storm_{storm_idx}/{stat}_{storm_idx}_{level}.csv') #locals()[f'mean_{var_name}_{level}']
                        if df.shape[0] > 0:  # Check if the csv is not empty
                            storm_series = df.loc[:, '0'].values
                            storm_data_y.append(storm_series)
                        else:
                            print(f"Storm {storm_idx} not found in {var_name} for level {level}")
                            storm_data_y.append(np.array([]))  # Append an empty array if the storm index does not exist
                except KeyError:
                    storm_data_y.append(np.array([]))  # Append an empty array if the variable is not found
            else:
                try:
                    for stat in stats:
                        df = pd.read_csv(f'{path}/data/{dataset}/{var_name}/storm_{storm_idx}/{stat}_{storm_idx}_{level}.csv') #locals()[f'mean_{var_name}_{level}']
                        if df.shape[0] > 0:  # Check if the csv is not empty
                            storm_series = df.loc[:, '0'].values
                            storm_data.append(storm_series)
                        else:
                            print(f"Storm {storm_idx} not found in {var_name} for level {level}")
                            storm_data.append(np.array([]))  # Append an empty array if the storm index does not exist
                except KeyError:
                    storm_data.append(np.array([]))  # Append an empty array if the variable is not found

            # Find the maximum length of time steps for the current storm
            #max_time_steps = max(len(series) for series in storm_data)

        # Pad the data to have the same length of time steps
        storm_data_padded = [np.pad(series, (0, max_time_steps - len(series)), 'constant', constant_values=np.nan) for series in storm_data]
        storm_data_padded_y = [np.pad(series, (0, max_time_steps - len(series)), 'constant', constant_values=np.nan) for series in storm_data_y]

        #storm_data_padded = []
        #for series in storm_data:
        #    padded_series = np.pad(series, (0, max_time_steps - len(series)), 'constant', constant_values=np.nan)
        #    storm_data_padded.append(padded_series)

        # Stack the variables together and add to the list
        data_list.append(np.stack(storm_data_padded, axis=1))
        data_list_y.append(np.stack(storm_data_padded_y, axis=1))

    # Convert the list to a 3D numpy array
    X_all_3d = np.stack(data_list, axis=0)
    y_all_3d = np.stack(data_list_y, axis=0)

    print("Shape of the X 3D ndarray:", X_all_3d.shape)
    print("Shape of the y 3D ndarray:", y_all_3d.shape)

    # check if storms in tracks_1h_EU are continuous with the column timestep
    index_storm_EU = []
    for i in range(0, 96):
        locals()[f'storm_{i+1}'] = pd.read_csv(f'{path}pre_processing/tracks/ALL_TRACKS/tracks_1h_EU/storm_{i+1}.csv')
        try:
            if locals()[f'storm_{i+1}']['timestep'].values.max() -locals()[f'storm_{i+1}']['timestep'].values.min() == len(locals()[f'storm_{i+1}'])-1:
                print(f'Storm {i} is continuous.')
                index_storm_EU.append(i)
            else:
                print(f'Storm {i} is not continuous.')
        except ValueError:
            print(f'Storm {i} is empty.')

    storm_index_test_valid = [0, 3, 4, 11, 13, 14, 17, 20, 25, 27, 28, 29, 31, 35, 53, 54, 57, 64, 69, 71, 75, 81, 85, 86, 87, 90, 92, 93, 95]
    storm_index_validation = [3, 4, 11, 17, 31, 35, 54, 86, 87, 92]
    storm_index_all = range(96)

    # remove index of storm in the test set from the validation set

    storm_index_test = [x for x in storm_index_test_valid if x not in storm_index_validation]

    # remove index of storm in the valdation set from the test set

    storm_index_validation = [x for x in storm_index_validation if x not in storm_index_test]

    # remove index of storm in the training set from the validation set and validation set

    storm_index_training = [x for x in storm_index_all if x not in storm_index_test_valid]

    print(storm_index_validation, storm_index_test)
    print(storm_index_training)

    storm_index_training = [x for x in storm_index_training if x in index_storm_EU]
    storm_index_validation = [x for x in storm_index_validation if x in index_storm_EU]
    storm_index_test = [x for x in storm_index_test if x in index_storm_EU]

    print("Validation indices:", storm_index_validation)
    print("Test indices:", storm_index_test)
    print("Training indices:", storm_index_training)

    # extract the data for the test set and the validation set
    #number_storms_EU =  range(len(index_storm_EU))

    X_test = X_all_3d[storm_index_test,:,:]
    X_validation = X_all_3d[storm_index_validation,:,:]
    X_train = X_all_3d[storm_index_training,:,:]

    y_test = y_all_3d[storm_index_test,:,:]
    y_validation = y_all_3d[storm_index_validation,:,:]
    y_train = y_all_3d[storm_index_training,:,:]

    return X_train, X_test, X_validation, y_train, y_test, y_validation

def square_EU(EU_border, tracks, choosen_directory, path):

    # now cut in the folder EU_border all the storms that are not in the EU

    for i in range(1, 97):
        storm = pd.read_csv(f'{path}pre_processing/tracks/ALL_TRACKS/{tracks}/storm_{i}.csv')
        rows = range(0,storm.shape[0])
        storm['center_lat'] = (storm['lat_north'] + storm['lat_south']) / 2
        storm['center_lon'] = np.zeros(storm.shape[0])
        for row in rows:
            if storm.loc[row,'lon_west'] > storm.loc[row,'lon_east']:
                storm['center_lon'][row] = storm['lon_west'][row] - 4
            else:
                storm['center_lon'][row] = storm['lon_east'][row] + 4
        # add a column names timestep
        storm['timestep'] = storm.index

        # Create a boolean mask for rows within EU borders NOT WORKING
        mask = (storm['center_lat'] >= EU_border['south'][0]) & \
               (storm['center_lat'] <= EU_border['north'][0]) & \
               (storm['center_lon'] >= EU_border['west'][0]) & \
               (storm['center_lon'] <= EU_border['east'][0])
        
        for row in rows:
            if storm['center_lat'][row] < EU_border['south'][0] or \
                storm['center_lat'][row] > EU_border['north'][0] or \
                storm['center_lon'][row] < EU_border['west'][0] and storm['center_lon'][row] > 180 or \
                storm['center_lon'][row] > EU_border['east'][0] and storm['center_lon'][row] < 180:
                #drop the row that are not in the EU
                    storm.drop(row, inplace=True)
        
        # Filter the DataFrame using the mask
        #storm = storm[mask]

        # drop the columns center_lat and center_lon
        storm.drop(['center_lat', 'center_lon'], axis=1, inplace=True)

        # Create the directory if it doesn't exist
        if not os.path.exists(f'{path}{choosen_directory}/{tracks}_EU/'):
            os.makedirs(f'{path}{choosen_directory}/{tracks}_EU/')

        # Save the filtered DataFrame to the same file or a new file
        storm.to_csv(f'{path}{choosen_directory}/{tracks}_EU/storm_{i}.csv', index=False)

        '''for j in range(0,len_storm):
            if storm['center_lat'][j] < EU_border['south'][0] or storm['center_lat'][j] > EU_border['north'][0] or storm['center_lon'][j] < EU_border['west'][0] or storm['center_lon'][j] > EU_border['east'][0]:
                #drop the row that are not in the EU
                storm.drop(storm.index, inplace=True)'''

        '''for index, row in storm.iterrows():
            if not EU_border['Latitude'].between(row['lat_south'], row['lat_north']).any() or not EU_border['Longitude'].between(row['lon_west'], row['lon_east']).any():
                print(f'storm_{i} is not in the EU')
                print(f'center_lat: {row["center_lat"]}')
                print(f'center_lon: {row["center_lon"]}')
            else:
                print(f'storm_{i} is in the EU')
                print(f'center_lat: {row["center_lat"]}')
                print(f'center_lon: {row["center_lon"]}')'''

def hourly_steps (factor,choosen_directory, path):
    for i in range(1,97):
        locals()['tracks_'+str(i)] = pd.read_csv(f'{path}pre_processing/tracks/ALL_TRACKS/tracks_3h/storm_{i}.csv')
        locals()['lon_east_'+str(i)] = locals()['tracks_'+str(i)]['lon_east']
        locals()['lat_north_'+str(i)] = locals()['tracks_'+str(i)]['lat_north']
        locals()['lon_west_'+str(i)] = locals()['tracks_'+str(i)]['lon_west']
        locals()['lat_south_'+str(i)] = locals()['tracks_'+str(i)]['lat_south']

    # interpolation between 2 points in a vector
    def interpolate_vector(data, factor):
        n = len(data)
        # X interpolation points. For factor=4, it is [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, ...]
        x = np.linspace(0, n - 1, (n - 1) * factor + 1)
        # Alternatively:
        # x = np.arange((n - 1) * factor + 1) / factor
        # X data points: [0, 1, 2, ...]
        xp = np.arange(n)
        # Interpolate
        return np.round(np.interp(x, xp, np.asarray(data)),6)

    # apply the interpolation to each longitude and latitude vectors

# for latitudes

    for i in range(1,97):
        locals()['lat_north_'+str(i)+'_interp'] = interpolate_vector(locals()['lat_north_'+str(i)], 3)
        locals()['lat_south_'+str(i)+'_interp'] = interpolate_vector(locals()['lat_south_'+str(i)], 3)

    # for longitudes 

    for i in range(1,97):

        locals()['lon_east_'+str(i)+'_interp'] = []

        for j in range(0,len(locals()['lon_east_'+str(i)])-1):

            if abs(locals()['lon_east_'+str(i)][j] - locals()['lon_east_'+str(i)][j+1]) > 300:
                if locals()['lon_east_'+str(i)][j] > locals()['lon_east_'+str(i)][j+1]:
                    delta_before_360 = 360 - locals()['lon_east_'+str(i)][j]
                    delta_after_360 = locals()['lon_east_'+str(i)][j+1]
                    sum_deltas = delta_before_360 + delta_after_360
                    delta_factor = sum_deltas / factor

                    locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j])
                    if (locals()['lon_east_'+str(i)][j]+delta_factor) > 360:
                        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+delta_factor-360)
                    else:
                        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+delta_factor)
                    if (locals()['lon_east_'+str(i)][j]+2*delta_factor) > 360:
                        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+2*delta_factor-360)
                    else:
                        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+2*delta_factor)
                else:
                    delta_before_360 = 360 - locals()['lon_east_'+str(i)][j+1]
                    delta_after_360 = locals()['lon_east_'+str(i)][j]
                    sum_deltas = delta_before_360 + delta_after_360
                    delta_factor = -sum_deltas / factor

                    locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j])
                    if (locals()['lon_east_'+str(i)][j]+delta_factor) < 0:
                        locals()['lon_east_'+str(i)+'_interp'].append(360+locals()['lon_east_'+str(i)][j]+delta_factor)
                    else:
                        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+delta_factor)
                    if (locals()['lon_east_'+str(i)][j]+2*delta_factor) < 0:
                        locals()['lon_east_'+str(i)+'_interp'].append(360+locals()['lon_east_'+str(i)][j]+2*delta_factor)
                    else:
                        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+2*delta_factor)

            else:

                delta_before_360 = locals()['lon_east_'+str(i)][j+1] - locals()['lon_east_'+str(i)][j]
                delta_factor = delta_before_360 / factor
                
                locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j])
                locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+delta_factor)
                locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][j]+2*delta_factor)

        locals()['lon_east_'+str(i)+'_interp'].append(locals()['lon_east_'+str(i)][len(locals()['lon_east_'+str(i)])-1])

    for i in range(1,97):

        locals()['lon_west_'+str(i)+'_interp'] = []

        for j in range(0,len(locals()['lon_west_'+str(i)])-1):

            if abs(locals()['lon_west_'+str(i)][j] - locals()['lon_west_'+str(i)][j+1]) > 300:
                if locals()['lon_west_'+str(i)][j] > locals()['lon_west_'+str(i)][j+1]:
                    delta_before_360 = 360 - locals()['lon_west_'+str(i)][j]
                    delta_after_360 = locals()['lon_west_'+str(i)][j+1]
                    sum_deltas = delta_before_360 + delta_after_360
                    delta_factor = sum_deltas / factor

                    locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j])
                    if (locals()['lon_west_'+str(i)][j]+delta_factor) > 360:
                        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+delta_factor-360)
                    else:
                        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+delta_factor)
                    if (locals()['lon_west_'+str(i)][j]+2*delta_factor) > 360:
                        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+2*delta_factor-360)
                    else:
                        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+2*delta_factor)
                else:
                    delta_before_360 = 360 - locals()['lon_west_'+str(i)][j+1]
                    delta_after_360 = locals()['lon_west_'+str(i)][j]
                    sum_deltas = delta_before_360 + delta_after_360
                    delta_factor = -sum_deltas / factor

                    locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j])
                    if (locals()['lon_west_'+str(i)][j]+delta_factor) < 0:
                        locals()['lon_west_'+str(i)+'_interp'].append(360+locals()['lon_west_'+str(i)][j]+delta_factor)
                    else:
                        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+delta_factor)
                    if (locals()['lon_west_'+str(i)][j]+2*delta_factor) < 0:
                        locals()['lon_west_'+str(i)+'_interp'].append(360+locals()['lon_west_'+str(i)][j]+2*delta_factor)
                    else:
                        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+2*delta_factor)

            else:

                delta_before_360 = locals()['lon_west_'+str(i)][j+1] - locals()['lon_west_'+str(i)][j]
                delta_factor = delta_before_360 / factor
                
                locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j])
                locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+delta_factor)
                locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][j]+2*delta_factor)

        locals()['lon_west_'+str(i)+'_interp'].append(locals()['lon_west_'+str(i)][len(locals()['lon_west_'+str(i)])-1])

    # round the values to 6 decimals

    for i in range(1,97):
        locals()['lon_east_'+str(i)+'_interp'] = [round(num, 6) for num in locals()['lon_east_'+str(i)+'_interp']]
        locals()['lon_west_'+str(i)+'_interp'] = [round(num, 6) for num in locals()['lon_west_'+str(i)+'_interp']]
        locals()['lat_north_'+str(i)+'_interp'] = [round(num, 6) for num in locals()['lat_north_'+str(i)+'_interp']]
        locals()['lat_south_'+str(i)+'_interp'] = [round(num, 6) for num in locals()['lat_south_'+str(i)+'_interp']]

        # combine the interpolated vectors into a dataframe

    for i in range(1,97):
        locals()['tracks_'+str(i)+'_interp'] = pd.DataFrame({'lon_east':locals()['lon_east_'+str(i)+'_interp'],'lon_west':locals()['lon_west_'+str(i)+'_interp'],'lat_south':locals()['lat_south_'+str(i)+'_interp'],'lat_north':locals()['lat_north_'+str(i)+'_interp']})

        # save the interpolated dataframes to csv files

            # Create the directory if it doesn't exist
    if not os.path.exists(f'{path}{choosen_directory}/tracks_1h/'):
        os.makedirs(f'{path}{choosen_directory}/tracks_1h/')


    for i in range(1,97):
        locals()['tracks_'+str(i)+'_interp'].to_csv(f'{path}{choosen_directory}/tracks_1h/storm_{i}.csv',index=False)














