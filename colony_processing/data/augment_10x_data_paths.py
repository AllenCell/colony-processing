def load_data_augment_sets():
    dataset_dict = {
        'AICS-7': [100],
        'AICS-54': [102],
        'AICS-13': [103],
        'AICS-53': [119],
        'AICS-23': [118],
        'AICS-11': [110],
        'AICS-57': [105],
        'AICS-58': [121],
        'AICS-12': [97],
        'AICS-10': [111],
        'AICS-24': [99],
        'AICS-22': [113],
    }
    return dataset_dict

dataset_dict = load_data_augment_sets()
proj_folder = r'\\allen\aics\microscopy\Data\fov_in_colony'

import pandas as pd
import os
import numpy as np

# augment data: 10X csv, fix file name errors (350000xxxx and 3500000xxxx)
for cell_line, datasets in dataset_dict.items():
    for dataset_id in datasets:
        df_check = pd.read_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_10x.csv'))
        for index, row in df_check.iterrows():

            pipeline = row['workflow'][2:-2]
            if pipeline == 'Pipeline 4.4':
                pipeline_folder = r'\\allen\aics\microscopy\PRODUCTION\PIPELINE_4_4'
            else:
                pipeline_folder = r'\\allen\aics\microscopy\PRODUCTION\PIPELINE_4_OptimizationAutomation'

            # if row['filename_10x'] is np.nan:  # remove this clause, 10X czi after upload format is different
            barcode_folder = os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']))
            zsd_folder = os.listdir(barcode_folder)
            for folder in zsd_folder:
                if folder.startswith('ZSD'):
                    possible_10x_folders = []
                    if os.path.exists(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10Xwellscan')):
                        possible_10x_folders.append(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10Xwellscan'))
                    if os.path.exists(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10XwellScan')):
                        possible_10x_folders.append(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10Xwellscan'))
                    if os.path.exists(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10Xwellscan', 'split')):
                        possible_10x_folders.append(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10Xwellscan', 'split'))
                    if os.path.exists(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10XwellScan', 'split')):
                        possible_10x_folders.append(os.path.join(r'\\allen\aics\microscopy\PRODUCTION', pipeline_folder, str(row['barcode']), folder, '10XwellScan', 'split'))

            for folder_10x in possible_10x_folders:
                files = os.listdir(folder_10x)
                for file in files:
                    if (str(row['barcode'])[-4:] in file) & (row['well'] in file):
                        df_check.loc[index, 'filename_10x'] = file
                        df_check.loc[index, 'filepath_10x'] = folder_10x[1:].replace('\\', '/') + '/' + file
                        break

            # per plate augments:
            #1. 3500000924
            if str(row['barcode']) == '3500000924':
                typo_barcode = '0922'
                for folder_10x in possible_10x_folders:
                    files = os.listdir(folder_10x)
                    for file in files:
                        if (typo_barcode in file) & (row['well'] in file):
                            df_check.loc[index, 'filename_10x'] = file
                            df_check.loc[index, 'filepath_10x'] = folder_10x[1:].replace('\\', '/') + '/' + file
                            break

            #2.3500001379
            if str(row['barcode']) == '3500001379':
                typo_barcode = '13679'
                for folder_10x in possible_10x_folders:
                    files = os.listdir(folder_10x)
                    for file in files:
                        if (typo_barcode in file) & (row['well'] in file):
                            df_check.loc[index, 'filename_10x'] = file
                            df_check.loc[index, 'filepath_10x'] = folder_10x[1:].replace('\\', '/') + '/' + file
                            break

        df_check = df_check[['barcode', 'well', 'workflow', 'filename_10x', 'filepath_10x']]
        df_check.to_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_10x.csv'))
