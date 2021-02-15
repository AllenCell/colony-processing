# Get imaging date info for wells

import pandas as pd
from lkaccess import LabKey
from variance_fov_colony.data.load_dataset import load_shape_space_csv

lk = LabKey(host='aics.corp.alleninstitute.org')

all_df = load_shape_space_csv()
all_well_id = all_df['WellId'].unique()

query_filter = ''
for well in all_well_id:
    query_filter += ';' + str(well)

lk = LabKey(host='aics.corp.alleninstitute.org')

query_wells = lk.select_rows_as_list(
    schema_name='microscopy',
    query_name='FOV',
    filter_array=[
        ('Objective', '100', 'eq'),
        ('WellId/WellId', query_filter[1:], 'in')
    ],
    columns=[
        'WellId',
        'FOVImageDate',
    ]
)
all_date = pd.DataFrame(query_wells)
df_image_date = all_date.groupby('WellId').min()

df_image_date.to_csv(r'\\allen\aics\microscopy\Data\fov_in_colony\fov_image_date.csv')
