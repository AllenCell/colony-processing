# merge csvs from all cell lines

import pandas as pd
import os
from variance_fov_colony.data.load_dataset import load_dataset

dataset_dict = load_dataset()
proj_folder = r'\\allen\aics\microscopy\Data\fov_in_colony'

df = pd.DataFrame()

for cell_line, datasets in dataset_dict.items():
    for dataset_id in datasets:
        pos_info = pd.read_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_position_in_colony.csv'))
        pos_info['cell_line'] = cell_line
        pos_info['dataset_id'] = dataset_id

        df = df.append(pos_info, ignore_index=True)

df.to_csv(os.path.join(proj_folder, 'all_position_in_colony.csv'))
