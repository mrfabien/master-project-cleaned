import xarray as xr

def open_monthly_nc(year, variable, months, way, level=0):
    try :
        datasets = [xr.open_dataset(f'{way}{variable}/ERA5_{year}-{month}_{variable}.nc') for month in months]
        if variable == 'geopotential' and level != 0:
            datasets = [dataset.sel(level=level) for dataset in datasets]
    except:
        print('Format of the raw variables must be {variable}/ERA5_{year}-{month}_{variable}.nc')
    return xr.concat(datasets, dim='time')

def process_data(year, way, variable)

    year_next = year + 1
    month_act = [10, 11, 12]
    month_next = [1, 2, 3]

    # Open and concatenate datasets
    if year == 1990:
        dataset_act = open_monthly_nc(str(year), variable, month_next, way, level)
        dataset_next = open_monthly_nc(str(year_next), variable, month_next, way, level)
        dataset = xr.concat([dataset_act, dataset_next], dim='time')
        dataset = dataset.chunk({'time': 10})
    elif year == 2021:
        dataset = open_monthly_nc(str(year), variable, month_next, way, level)
    else:
        dataset_act = open_monthly_nc(str(year), variable, month_act, way, level)
        dataset_next = open_monthly_nc(str(year_next), variable, month_next, way, level)
        dataset = xr.concat([dataset_act, dataset_next], dim='time')
        dataset = dataset.chunk({'time': 10})

    return dataset