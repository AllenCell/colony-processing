# Merge fov position in colony with cell meta csv (from Matheus/Jianxu)

import pandas as pd

from variance_fov_colony.data.load_dataset import load_fov_position_flag_csv, load_shape_space_csv

df_shape = load_shape_space_csv()
print(len(df_shape))

df_shape = df_shape[[x for x in df_shape.columns.values.tolist() if 'meta' not in x]]

df_colony = load_fov_position_flag_csv()

df_colony = df_colony.rename(columns={# 'FOVId': 'fov_id',
                                      'position_outside_overview': 'meta_fov_outside_overview',
                                      'x_px': 'meta_fov_xcoord',
                                      'y_px': 'meta_fov_ycoord',
                                      'dist_um': 'meta_fov_edgedist',
                                      'dist_center': 'meta_colony_dist_center',
                                      'fov_in_colony_touch_boundary':  'meta_colony_touching_boundary',
                                      'sum_colony_area': 'meta_colony_area_sum',
                                      'num_colonies': 'meta_colony_count',
                                      'area_colony': 'meta_colony_area',
                                      'centroid_colony': 'meta_colony_coords',
                                      'colony_dist_from_well_center': 'meta_colony_dist_center',
                                      'bad_10x_seg': 'meta_plate_bad_segmentation',
                                      'confluency': 'meta_confluency'})

df_colony = df_colony[['FOVId',
                       'meta_fov_outside_overview',
                       'meta_fov_xcoord',
                       'meta_fov_ycoord',
                       'meta_fov_edgedist',
                       'meta_colony_touching_boundary',
                       'meta_colony_area_sum',
                       'meta_colony_count',
                       'meta_colony_area',
                       'meta_colony_coords',
                       'meta_colony_dist_center',
                       'meta_plate_bad_segmentation',
                       'meta_confluency']
]

df = df_shape.merge(df_colony, left_on=['FOVId'], right_on=['FOVId'])
print(len(df))
df.to_csv(r'\\allen\aics\microscopy\Data\fov_in_colony\all_cell_meta.csv')
