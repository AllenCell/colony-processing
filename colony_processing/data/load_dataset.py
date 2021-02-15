# load dataset
import os
import pandas as pd
from lkaccess import LabKey

def load_dataset():
    dataset_dict = {
        'AICS-5': [120],
        'AICS-7': [100],
        'AICS-10': [111],
        'AICS-11': [110],
        'AICS-12': [97],
        'AICS-13': [103],
        'AICS-14': [104],
        'AICS-16': [98],
        'AICS-17': [117],
        'AICS-23': [118],
        'AICS-24': [99],
        'AICS-22': [113],
        'AICS-25': [112],
        'AICS-32': [116],
        'AICS-33': [114],
        'AICS-40': [115],
        'AICS-46': [101],
        'AICS-53': [119],
        'AICS-54': [102],
        'AICS-57': [105],
        'AICS-58': [121],
        'AICS-61': [106],
        'AICS-68': [108],
        'AICS-69': [107],
        'AICS-94': [109],
    }

    return dataset_dict

def load_dataset_old():
    dataset_dict = {
        'AICS-5': [35],
        'AICS-7': [5],
        'AICS-10': [14, 88],
        'AICS-11': [3, 87],
        'AICS-12': [4, 86],
        'AICS-13': [12, 93],
        'AICS-14': [11],
        'AICS-16': [6],
        'AICS-17': [10],
        'AICS-22': [2, 90],
        'AICS-23': [16],
        'AICS-24': [43, 89],
        'AICS-25': [15],
        'AICS-32': [8],
        'AICS-33': [40],
        'AICS-40': [41],
        'AICS-46': [94],
        'AICS-53': [9],
        'AICS-54': [25],
        'AICS-57': [44],
        'AICS-58': [7],
        'AICS-61': [83],
        'AICS-68': [95],
        'AICS-69': [84],
        'AICS-94': [91, 92]

    }

    return dataset_dict


def load_dataset_some():
    dataset_dict = {
        'AICS-12': [4, 86],
        'AICS-16': [6],
        'AICS-24': [89],
        'AICS-7': [5],
        'AICS-46': [],
        'AICS-54': [25],
        'AICS-13': [12],
        'AICS-14': [11],
        'AICS-57': [44],
        'AICS-11': [3, 87],
        'AICS-10': [14, 88],
        'AICS-25': [15],
        'AICS-22': [2, 90],
        'AICS-32': [8],
        'AICS-17': [10],
        'AICS-23': [16],
        'AICS-53': [9],
        'AICS-5': [35],
        'AICS-58': [7]
    }
    return dataset_dict


def load_fov_position_csv(system='Windows', proj_folder='/allen/aics/microscopy/Data/fov_in_colony', file_name='all_position_in_colony.csv'):
    if system == 'Windows':
        path = os.path.join('\\' + proj_folder, file_name)
    else:
        path = os.path.join(proj_folder, file_name)
    df = pd.read_csv(path)

    return df


def load_shape_space_csv(system='Windows', filepath='/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging/expand/manifest.csv'):

    if system == 'Windows':
        path = '\\' + filepath
    else:
        path = filepath

    df = pd.read_csv(path)

    return df


def load_fov_position_flag_csv(system='Windows', proj_folder='/allen/aics/microscopy/Data/fov_in_colony', file_name='all_position_in_colony_flag.csv'):

    if system == 'Windows':
        path = os.path.join('\\' + proj_folder, file_name)
    else:
        path = os.path.join(proj_folder, file_name)

    df = pd.read_csv(path)

    return df


def load_cell_meta_csv(system='Windows', proj_folder='/allen/aics/microscopy/Data/fov_in_colony', file_name='all_cell_meta.csv'):
    if system == 'Windows':
        path = os.path.join('\\' + proj_folder, file_name)
    else:
        path = os.path.join(proj_folder, file_name)

    df = pd.read_csv(path)

    return df


def load_imaging_mode():
    lk = LabKey(host='aics.corp.alleninstitute.org')

    query_wells = lk.select_rows_as_list(
        schema_name='microscopy',
        query_name='WellImagingModeJunction',
        columns=[
            'WellId',
            'WellId/PlateId/BarCode',
            'WellId/WellName/Name',
            'ImagingModeId/Name'
        ]
    )
    df_well_mode = pd.DataFrame(query_wells)

    df_well_mode = df_well_mode.rename(columns={'ImagingModeId/Name': 'ImagingMode',
                                                'WellId/WellName/Name': 'WellName',
                                                'WellId/PlateId/BarCode': 'BarCode'})

    return df_well_mode


def load_passaging_info(system='Windows', proj_folder='/allen/aics/microscopy/Data/fov_in_colony', file_name='passage_info_with_calc.csv'):
    if system == 'Windows':
        if system == 'Windows':
            path = os.path.join('\\' + proj_folder, file_name)
        else:
            path = os.path.join(proj_folder, file_name)

        df = pd.read_csv(path)

        return df


def load_image_date(system='Windows', proj_folder='/allen/aics/microscopy/Data/fov_in_colony', file_name='fov_image_date.csv'):
    if system == 'Windows':
        if system == 'Windows':
            path = os.path.join('\\' + proj_folder, file_name)
        else:
            path = os.path.join(proj_folder, file_name)

        df = pd.read_csv(path)
        df['FOVImageDate'] = pd.to_datetime(df['FOVImageDate'])
        return df
