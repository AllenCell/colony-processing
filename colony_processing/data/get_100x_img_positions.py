from lkaccess import LabKey
import pandas as pd
import numpy as np
import os
from ast import literal_eval
from variance_fov_colony.data.load_dataset import load_dataset

proj_folder = '\\' + '/allen/aics/microscopy/Data/fov_in_colony'

# Connect to host on labkey
lk = LabKey(host="aics.corp.alleninstitute.org")


def get_dataset(id):
    """
    Retrieves a dataset from labkey
    Parameters
    ----------
    id          dataset id from labkey

    Returns
    -------
    df          dataframe with entries in a dataset, with columns `barcode`, `filename`, `workflow`, `well`
    """
    my_results = lk.select_rows_as_list(
        schema_name='datahandoff',
        query_name='DatasetFOVJunctionCustom',
        view_name='Microscopy',
        filter_array=[
            ('DatasetId', id, 'eq')
        ],
        columns=[
            'FOVId',
            'FOVImageDate',
            'QCStatusId',
            'WellId/PlateId/Workflow/Name',
            'WellId/PlateId/BarCode',
            'SourceImageFileId/Filename',
            'WellId/Wellname/Name'
        ]
    )

    df = pd.DataFrame(my_results)

    df = df.rename(columns={
        'WellId/PlateId/BarCode': 'barcode',
        'SourceImageFileId/Filename': 'filename',
        'WellId/PlateId/Workflow/Name': 'workflow',
        'WellId/Wellname/Name': 'well'
    }
    )

    return df


def get_100X_position(img_name):
    """
    Retrieves the FOV stage position from labkey per image
    Parameters
    ----------
    img_name            image name that can be used to query on fms File table

    Returns
    -------
    df                  dataframe with of the FOV with `raw_ome_filename`, `FOVId/CenterX`, `FOVId/CenterY`
    """
    my_results = lk.select_rows_as_list(
        schema_name='fms',
        query_name='File',
        filter_array=[
            ('Filename', img_name, 'eq'),
        ],
        columns=[
            'FOVId/CenterX',
            'FOVId/CenterY',
            'Filename'
        ]
    )
    df = pd.DataFrame(my_results)
    df = df.rename(columns={'Filename': 'raw_ome_filename'})
    if len(df) == 0:
        return None
    else:
        return df

# Query from labkey, all FOV stage position
my_results = lk.select_rows_as_list(
    schema_name='fms',
    query_name='File',
    filter_array=[
        ('Filename', '100', 'contains'),
        ('Filename', '.ome', 'contains')
    ],
    columns=[
        'FOVId/CenterX',
        'FOVId/CenterY',
        'FOVId/CenterY',
        'Filename'
    ]
)
df = pd.DataFrame(my_results)
df = df.rename(columns={'Filename': 'raw_ome_filename'})

dataset_dict = load_dataset()

# loop for each cell line/dataset to retrieve 100X FOV stage positions
for cell_line, datasets in dataset_dict.items():
    for dataset_id in datasets:
        print(cell_line, dataset_id)
        # get dataset
        df_100X = get_dataset(id=dataset_id)
        df_100X['workflow'] = df_100X['workflow'].astype(str)
        # identify pipeline
        # pipeline = literal_eval(df_100X['workflow'].astype(str).unique()[0])[0]

        df_100X_4 = df_100X.loc[df_100X['workflow'] != "['Pipeline 4.4']"]
        df_100X_4_4 = df_100X.loc[df_100X['workflow'] == "['Pipeline 4.4']"]

        if len(df_100X_4_4) > 0:
            df_100X_4_4['raw_filename'] = ''
            for index, row in df_100X_4_4.iterrows():
                raw_file = row['filename'].replace('_aligned_cropped', '')
                df_100X_4_4.loc[index, 'raw_filename'] = raw_file

            for index, row in df_100X_4_4.iterrows():
                if (row['raw_filename'].startswith('3500002359')) | (row['raw_filename'].startswith('3500002365')):
                    df_100X_4_4.loc[index, 'raw_filename'] = row['raw_filename'].replace('-P', '_P')
                if '-aligned_cropped' in row['raw_filename']:
                    if row['raw_filename'].startswith('3500002365') is False:
                        df_100X_4_4.loc[index, 'raw_filename'] = row['raw_filename'].replace('-aligned_cropped', '')

            keep = df_100X_4_4.merge(df, how='left', left_on=['raw_filename'], right_on=['raw_ome_filename'])
            merge_4_4 = keep.loc[keep['FOVId/CenterY'].str.len() != 0]
            for index, row in merge_4_4.iterrows():
                if row['FOVId/CenterX'] is not np.nan:
                    # store data as a float
                    for column in ['FOVId/CenterX', 'FOVId/CenterY']:
                        merge_4_4.loc[index, column] = row[column][0]
            merge_4_4 = merge_4_4.rename(columns={'file_name_100x': 'file_name'})

            merge_4_4 = merge_4_4[['workflow', 'filename', 'QCStatusId', 'well', 'FOVId', 'barcode',
                           'FOVId/CenterX', 'FOVId/CenterY']]
        else:
            merge_4_4 = None

        if len(df_100X_4) > 0:
            prod_folder = '\\' + '/allen/aics/microscopy/PRODUCTION/PIPELINE_4_OptimizationAutomation'.replace('/', '\\')

            barcodes = df_100X_4['barcode'].unique()
            meta_100x = pd.DataFrame()
            for barcode in barcodes:

                # Augment some barcode path
                to_augment = [
                    '3500000883', '3500001827', '3500001835', '3500001847', '3500001845', '3500001830',
                    '3500001868', '3500001859', '3500002003', '3500001856', '3500001872', '3500001888',
                    '3500000815'
                ]

                # Get fov img position from acquisition_data.csv in plate folder
                system = [x for x in os.listdir(os.path.join(prod_folder, barcode)) if x.startswith('ZSD')][0]

                for folder_100x in ['100X_zstack', '100XA_zstack', '100XB_zstack']:
                    if os.path.exists(os.path.join(prod_folder, barcode, system, folder_100x, 'acquisition_data.csv')):
                        data = pd.read_csv(os.path.join(prod_folder, barcode, system, folder_100x, 'acquisition_data.csv'))

                        # Rename columns to match database
                        data = data.rename(columns={'file_name::file_name': 'file_name_100x',
                                                    'stage_x::stage_x!!R': 'FOVId/CenterX',
                                                    'stage_y::stage_y!!R': 'FOVId/CenterY',
                                                    }
                                           )
                        for index, row in data.iterrows():
                            if index > 0:
                                append_row = row.copy()

                                if barcode in to_augment:
                                    file_name = row['file_name_100x']
                                    append_row['file_name_100x'] = file_name.replace('350', '35')

                                append_row['filepath_100x'] = prod_folder + '/' + barcode + '/' + system + '/' + \
                                                              folder_100x + '/' + row['file_name_100x']

                                meta_100x = meta_100x.append(append_row, ignore_index=True)

            meta_100x['filename'] = meta_100x['file_name_100x'].str.replace('.czi', '.ome.tiff')
            merge_4 = df_100X_4.merge(meta_100x, on='filename')

            merge_4 = merge_4.rename(columns={'file_name_100x': 'file_name'})

            merge_4 = merge_4[['workflow', 'filename', 'QCStatusId', 'well', 'FOVId', 'barcode',
                           'FOVId/CenterX', 'FOVId/CenterY']]
        else:
            merge_4 = None

        if (merge_4 is not None) & (merge_4_4 is not None):
            all = merge_4.append(merge_4_4, ignore_index=True)
        elif merge_4 is not None:
            all = merge_4.copy()
        else:
            all = merge_4_4.copy()

        for barcode in df_100X['barcode'].unique():
            if barcode not in all['barcode'].unique():
                print(barcode)
        # print(len(df_100X) - len(all))
        # print(len(all))

        # Get 4.4 FOV positions from fms
        # if '4.4' in pipeline:
        #     print('in pipeline 4.4')
        #     for index, row in df_100X.iterrows():
        #         raw_file = row['filename'].replace('_aligned_cropped', '')
        #         df_100X.loc[index, 'raw_filename'] = raw_file
        #
        #     # edit df_100X to ingest specific data
        #     for index, row in df_100X.iterrows():
        #         if (row['raw_filename'].startswith('3500002359')) | (row['raw_filename'].startswith('3500002365')):
        #             df_100X.loc[index, 'raw_filename'] = row['raw_filename'].replace('-P', '_P')
        #         if '-aligned_cropped' in row['raw_filename']:
        #             if row['raw_filename'].startswith('3500002365') is False:
        #                 df_100X.loc[index, 'raw_filename'] = row['raw_filename'].replace('-aligned_cropped', '')
        #
        #     keep = df_100X.merge(df, how='left', left_on=['raw_filename'], right_on=['raw_ome_filename'])
        #     merge = keep.loc[keep['FOVId/CenterY'].str.len() != 0]
        #
        #     for index, row in merge.iterrows():
        #         if row['FOVId/CenterX'] is not np.nan:
        #             # store data as a float
        #             for column in ['FOVId/CenterX', 'FOVId/CenterY']:
        #                 merge.loc[index, column] = row[column][0]
        #
        # # Get fov positions from pipeline earlier than 4.4 from the images
        # else:
        #     prod_folder = '\\' + '/allen/aics/microscopy/PRODUCTION/PIPELINE_4_OptimizationAutomation'.replace('/', '\\')
        #
        #     barcodes = df_100X['barcode'].unique()
        #     meta_100x = pd.DataFrame()
        #     for barcode in barcodes:
        #         # Get fov img position from acquisition_data.csv in plate folder
        #         system = [x for x in os.listdir(os.path.join(prod_folder, barcode)) if x.startswith('ZSD')][0]
        #
        #         for folder_100x in ['100X_zstack', '100XA_zstack', '100XB_zstack']:
        #             if os.path.exists(os.path.join(prod_folder, barcode, system, folder_100x, 'acquisition_data.csv')):
        #                 data = pd.read_csv(os.path.join(prod_folder, barcode, system, folder_100x, 'acquisition_data.csv'))
        #
        #                 # Rename columns to match database
        #                 data = data.rename(columns={'file_name::file_name': 'file_name_100x',
        #                                             'stage_x::stage_x!!R': 'FOVId/CenterX',
        #                                             'stage_y::stage_y!!R': 'FOVId/CenterY',
        #                                             }
        #                                    )
        #                 for index, row in data.iterrows():
        #                     if index > 0:
        #                         append_row = row.copy()
        #                         append_row['filepath_100x'] = prod_folder + '/' + barcode + '/' + system + '/' + \
        #                                                       folder_100x + '/' + row['file_name_100x']
        #
        #                         meta_100x = meta_100x.append(append_row, ignore_index=True)
        #
        #     meta_100x['filename'] = meta_100x['file_name_100x'].str.replace('.czi', '.ome.tiff')
        #     merge = df_100X.merge(meta_100x, on='filename')
        #
        # merge = merge.rename(columns={'file_name_100x': 'file_name'})
        #
        # merge = merge[['workflow', 'filename', 'QCStatusId', 'well', 'FOVId', 'barcode',
        #                'FOVId/CenterX', 'FOVId/CenterY']]

        # merge.to_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_100x_position.csv'))
        all.to_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_100x_position.csv'))
