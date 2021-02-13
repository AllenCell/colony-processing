# merge features for 100X fov position in a colony

import pandas as pd
import os
import numpy as np
from variance_fov_colony.data.load_dataset import load_dataset

proj_folder = '\\' + '/allen/aics/microscopy/Data/fov_in_colony'.replace('/', '\\')
dataset_dict = load_dataset()

for cell_line, datasets in dataset_dict.items():
    for dataset_id in datasets:
        # Merge 10X info
        meta_10x = pd.read_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_10x_meta.csv'))
        per_colony = pd.read_csv(os.path.join(proj_folder, 'colony_seg', 'output_per_colony_' + str(cell_line.split('-')[-1]) + '_' + str(dataset_id) + '.csv'))

        # add background row for each image
        images = per_colony['file_name'].unique()
        for img in images:
            row = {}
            row['colony_num'] = 0
            row['file_name'] = img

            per_colony = per_colony.append(row, ignore_index=True)

        per_well = pd.read_csv(os.path.join(proj_folder, 'colony_seg', 'output_per_well_' + str(cell_line.split('-')[-1]) + '_' + str(dataset_id) + '.csv'))

        colony_info = per_colony.merge(per_well, on='file_name', how='outer')
        meta_colony = colony_info.merge(meta_10x, left_on=['file_name'], right_on=['filename_10x'])

        meta_colony = meta_colony.rename(columns={
            'stage_x::stage_x!!R': 'stage_x_10x',
            'stage_y::stage_y!!R': 'stage_y_10x',
            'acquisition_date::acquisition_date': 'acquisition_date',
            'acquisition_year::acquisition_year': 'acquisition_year',
            'acquisition_time::acquisition_time': 'acquisition_time',
            'colony_num': 'colony_number'
        }
        )

        meta_colony = meta_colony.fillna(value={'colony_number': 0})

        meta_colony = meta_colony[
            [
                'area_colony', 'centroid_colony', 'colony_number', 'dist_center',
                'confluency', 'max_colony_area', 'max_dist', 'mean_colony_area', 'mean_dist',
                'median_colony_area', 'median_dist', 'min_colony_area', 'min_dist',
                'num_colonies', 'std_colony_area', 'std_dist', 'sum_colony_area',
                'barcode', 'well',
                'stage_x_10x', 'stage_y_10x',
                'acquisition_date', 'acquisition_year', 'acquisition_time'
            ]
        ]

        # Merge 100X info
        meta_100x = pd.read_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_position_info_100x.csv'))
        meta_100x = meta_100x.dropna()

        # Merge 10X with 100X
        df_cell_line = meta_100x.merge(meta_colony, on=['barcode', 'well', 'colony_number'], how='left')

        df_cell_line = df_cell_line[
            [
                'FOVId', 'barcode', 'colony_number', 'dist_px', 'dist_um',
                'file_name_100x', 'file_name_10x', 'path_overlay', 'qc', 'well', 'x_px', 'y_px',
                'area_colony', 'centroid_colony', 'dist_center', 'confluency', 'max_colony_area',
                'max_dist', 'mean_colony_area', 'mean_dist',
                'median_colony_area', 'median_dist', 'min_colony_area', 'min_dist',
                'num_colonies', 'std_colony_area', 'std_dist', 'sum_colony_area',
                'stage_x_10x', 'stage_y_10x', 'acquisition_date', 'acquisition_year',
                'acquisition_time'
            ]
        ]

        # save csv
        df_cell_line.to_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_position_in_colony.csv'))
