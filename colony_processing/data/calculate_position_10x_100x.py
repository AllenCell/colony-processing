import math
import numpy as np
import os

import pandas as pd
from scipy import ndimage
from skimage import io

from variance_fov_colony.data.load_dataset import load_dataset

# calculates the position of an 100X fov in a colony
um_per_px = 1.08
seg_scaling_factor = 4
img_10x_reserve = '/allen/aics/microscopy/Data/fov_in_colony/colony_seg'

all_10x_imgs = os.listdir(img_10x_reserve)

all_csvs = '/allen/aics/microscopy/Data/fov_in_colony'
dataset_dict = load_dataset()

cell_lines = ['AICS-11']
datasets = [87]
stage_x_name = 'FOVId/CenterX'
stage_y_name = 'FOVId/CenterY'

# for cell_line, datasets in dataset_dict.items():
for cell_line in cell_lines:
    for dataset_id in datasets:
        count = 1

        # retrieve corresponding 10x and 100x csv
        df_10x = pd.read_csv(os.path.join(all_csvs, cell_line + '_' + str(dataset_id) + '_10x_meta.csv'))
        df_100x = pd.read_csv(os.path.join(all_csvs, cell_line + '_' + str(dataset_id) + '_100x_position.csv'))

        df = pd.DataFrame()

        for index, row in df_100x.iterrows():
            file_name = row['filename']
            print('processing ' + file_name)
            print(str(count) + ' out of ' + str(len(df_100x)))
            count += 1
            if file_name is not np.nan:
                plate = file_name.split('_')[0]

                if (row[stage_x_name] is not np.nan) & (row[stage_y_name] is not np.nan):
                    stage_x_100 = row[stage_x_name]
                    stage_y_100 = row[stage_y_name]

                    well = row['well']
                    new_row = {}
                    if (math.isnan(stage_x_100) == False) & (math.isnan(stage_y_100) == False):
                        img_10x_name = None
                        img_10x = df_10x.loc[(df_10x['barcode'] == int(plate)) & (df_10x['well'] == well), 'filename_10x'].values.tolist()
                        if len(img_10x) > 0:
                            img_10x_name = img_10x[0].replace('.czi', '_seg.tiff')

                        if img_10x_name is not None:

                            # get 10X stage position
                            stage_x_10 = df_10x.loc[(df_10x['barcode'] == int(plate)) & (df_10x['well'] == well), 'stage_x::stage_x!!R'].values.tolist()[0]
                            stage_y_10 = df_10x.loc[(df_10x['barcode'] == int(plate)) & (df_10x['well'] == well), 'stage_y::stage_y!!R'].values.tolist()[0]

                            # read segmentation image
                            seg = io.imread(os.path.join(img_10x_reserve, img_10x_name))
                            overlay = io.imread(os.path.join(img_10x_reserve, img_10x_name.replace('_seg.tiff', '_overlay.png')))
                            img_h_10 = seg.shape[0]
                            img_w_10 = seg.shape[1]

                            # calculate diff in x, diff in y in pixel
                            y_px = int(int(img_h_10 / 2) + (stage_y_100 - stage_y_10) / (seg_scaling_factor * um_per_px))
                            x_px = int(int(img_w_10 / 2) + (stage_x_100 - stage_x_10) / (seg_scaling_factor * um_per_px))

                            if (y_px < 0) | (y_px >= img_h_10) | (x_px < 0) | (x_px >= img_w_10) :
                                # position is outside of image, something wrong
                                row = {'file_name_100x': file_name,
                                       'file_name_10x': img_10x_name,
                                       'qc': 'fail'}
                            else:
                                overlay[y_px - 10:y_px + 10, x_px - 10:x_px + 10] = [255, 255, 255]
                                io.imsave(os.path.join(img_10x_reserve, 'position_10x', file_name.split('.czi')[0] + '_overlay_10x.png'),
                                          overlay)

                                # get nearest distance to 'empty' space
                                dt = ndimage.distance_transform_edt(seg)
                                dist_px = dt[y_px, x_px]
                                dist_um = dist_px * um_per_px * seg_scaling_factor
                                colony_num = seg[y_px, x_px]

                                new_row = {'file_name_100x': file_name,
                                           'file_name_10x': img_10x_name,
                                           'qc': 'pass',
                                           'path_overlay': os.path.join(img_10x_reserve, 'position_10x', file_name.split('.czi')[0] + '_overlay_10x.png'),
                                           'dist_um': dist_um,
                                           'dist_px': dist_px,
                                           'colony_number': colony_num,
                                           'y_px': y_px,
                                           'x_px': x_px,
                                           'barcode': plate,
                                           'well': row['well'],
                                           'FOVId': row['FOVId']}

                            df = df.append(new_row, ignore_index=True)
        df.to_csv(os.path.join(all_csvs, cell_line + '_' + str(dataset_id) + '_position_info_100x.csv'))
