# add flag/qc to colony position info

from ast import literal_eval
import os
import numpy as np

import pandas as pd
from skimage.io import imread

from variance_fov_colony.utilities.utilities import create_circular_mask
from variance_fov_colony.data.load_dataset import load_fov_position_csv

def flag_bad_overview_segmentation(df, confluency_threshold):
    """

    Parameters
    ----------
    df
    confluency_threshold

    Returns
    -------

    """
    df['bad_10x_seg'] = False
    df.loc[df['confluency'] < confluency_threshold[0], 'bad_10x_seg'] = True
    df.loc[df['confluency'] > confluency_threshold[1], 'bad_10x_seg'] = True
    return df


def flag_colony_touching_overview_boundary(df, seg_folder, seg_end_string='_seg.tiff'):
    """

    Parameters
    ----------
    df
    seg_folder
    seg_end_string

    Returns
    -------

    """
    df['fov_in_colony_touch_boundary'] = False
    df['seg_shape'] = ''
    files_10x = df['file_name_10x'].unique()
    df_10x = pd.DataFrame()
    for file in files_10x:
        seg = file.replace('.czi', seg_end_string)
        seg_img = imread(os.path.join(seg_folder, seg))
        img_height, img_width = seg_img.shape
        mask = create_circular_mask(seg_img.shape, radius=(img_height / 2.1) - 35)
        fail_colony_img = seg_img * mask

        barcode = df.loc[df['file_name_10x'] == file, 'barcode'].values.tolist()[0]
        # barcode = seg.split('_')[0]
        # well = seg.split('_seg.tiff')[0].split('_')[-1]
        well = df.loc[df['file_name_10x'] == file, 'well'].values.tolist()[0]
        if len(well) > 3:
            well = seg.split('_seg.tiff')[0].split('-')[-1]

        row = {}
        row['barcode'] = int(barcode)
        row['well'] = well
        row['colony_num_touch_boundary'] = list(np.unique(fail_colony_img).astype(int))
        row['mask_radius'] = (img_height / 2.1) - 20.
        row['seg_shape'] = seg_img.shape
        row['file_name_10x'] = seg

        df_10x = df_10x.append(row, ignore_index=True)

    for index, row in df.iterrows():
        colony_num = row['colony_number']

        if len(df_10x.loc[(df_10x['file_name_10x'] == row['file_name_10x']), 'colony_num_touch_boundary'].values.tolist()) > 0:
            colony_list = df_10x.loc[(df_10x['file_name_10x'] == row['file_name_10x']), 'colony_num_touch_boundary'].values.tolist()[0]
        else:
            colony_list = []

        if colony_num in colony_list:
            df.loc[index, 'fov_in_colony_touch_boundary'] = True
        df.at[index, 'seg_shape'] = df_10x.loc[(df_10x['file_name_10x'] == row['file_name_10x']), 'seg_shape'].values.tolist()[0]
        df.loc[index, 'mask_radius'] = df_10x.loc[(df_10x['file_name_10x'] == row['file_name_10x']), 'mask_radius'].values.tolist()[0]

    return df


def flag_position_outside_of_overview(df):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    df['position_outside_overview'] = False
    for index, row in df.iterrows():
        seg_shape = row['seg_shape']
        radius = row['mask_radius']

        mask = create_circular_mask(seg_shape, radius=radius)
        if (int(row['y_px']) >= mask.shape[0]) | (int(row['x_px']) >= mask.shape[1]):
            df.loc[index, 'position_outside_overview'] = True
        else:
            if mask[(int(row['y_px']), int(row['x_px']))]:
                df.loc[index, 'position_outside_overview'] = True
    return df


def flag_specific_wells(df):
    to_flag = {
        3500000981: ['E4', 'E5', 'E6', 'E7', 'E8', 'F4', 'F5', 'F6', 'F7', 'F8'],
        3500000971: ['F4', 'F5'],
        3500001079: ['E6'],
        3500001080: ['E4'],
        3500001126: ['E8'],
        3500001251: ['E8', 'F6'],
        3500001449: ['E6'],
        3500002669: ['C2', 'C3', 'C9'],
        3500003121: ['D3', 'D4', 'D5', 'D8', 'E5'],
        3500003720: ['D2', 'D10']
    }

    for barcode, wells in to_flag.items():
        for well in wells:
            df.loc[(df['barcode'] == barcode) & (df['well'] == well), 'bad_10x_seg'] = True

    return df

df = load_fov_position_csv()

df['colony_number'] = df['colony_number'].astype(int)

# flag bad overview segmentation
df = flag_bad_overview_segmentation(df, confluency_threshold=(0.1, 0.93))
df = flag_specific_wells(df)

# flag colonies touching edge of overview
seg_folder = r'\\allen\aics\microscopy\Data\fov_in_colony\colony_seg'
df = flag_colony_touching_overview_boundary(df, seg_folder)

# flag positions outside of the well captured
df = flag_position_outside_of_overview(df)

df.to_csv(r'\\allen\aics\microscopy\Data\fov_in_colony\all_position_in_colony_flag.csv')
