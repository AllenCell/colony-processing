from lkaccess import LabKey
import pandas as pd
import os
from ast import literal_eval
from variance_fov_colony.data.load_dataset import load_dataset
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


def get_10X_img(barcode, well):
    """
    From information on barcode and well, find the 10x image from fms
    Parameters
    ----------
    barcode             plate barcode
    well                well

    Returns
    -------
    a row containing the 10X img info
    """
    my_results = lk.select_rows_as_list(
        schema_name='fms',
        query_name='File',
        filter_array=[
            ('Filename', barcode, 'contains'),
            ('Filename', '10X', 'contains'),
            ('Filename', well, 'contains'),
            ('Filename', 'czi', 'contains')
        ]
    )
    df = pd.DataFrame(my_results)

    if len(df) == 0:
        return None
    elif len(df) == 1:
        df = df[['Filename', 'LocalFilePath']]
        return df
    else:
        print('more than one row')
        for index, row in df.iterrows():
            if 'Scene' not in row['Filename']:
                return df.iloc[[index]]
                break


dataset_dict = load_dataset()

proj_folder = r'\\allen\aics\microscopy\Data\fov_in_colony'

for cell_line, datasets in dataset_dict.items():
    for dataset_id in datasets:
        # retrieves 100X dataset
        df_100X = get_dataset(id=dataset_id)

        # identify all 10X images used to collect the 100X dataset
        df_check = df_100X[['barcode', 'well']].drop_duplicates()
        workflow_col = df_100X.loc[df_check.index, 'workflow']
        df_check = pd.merge(df_check, workflow_col, left_index=True, right_index=True)

        df_check['filename_10x'] = None
        df_check['filepath_10x'] = None
        count = 0

        # loop for the dataframe to check of the filepath exists, if not, add back in
        for index, row in df_check.iterrows():
            # identify pipeline information to set directory where to find file
            pipeline = row['workflow'][0]
            if '4.4' in pipeline:
                pipeline_folder = 'PIPELINE_4_4'
            else:
                pipeline_folder = 'PIPELINE_4_OptimizationAutomation'

            print('processing ' + str(count) + ' out of ' + str(len(df_check)))
            row_10x = get_10X_img(row['barcode'], row['well'])
            if row_10x is not None:
                if len(row_10x) == 1:
                    df_check.loc[index, 'filename_10x'] = row_10x['Filename'].values.tolist()[0]
                    df_check.loc[index, 'filepath_10x'] = row_10x['LocalFilePath'].values.tolist()[0]
            count += 1

            if (type(row['filepath_10x']) is not str) or (row['filepath_10x'] is None):
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
                        if (str(row['barcode']) in file) & (row['well'] in file):
                            df_check.loc[index, 'filename_10x'] = file
                            df_check.loc[index, 'filepath_10x'] = folder_10x[1:].replace('\\', '/') + '/' + file
                            break

        df_check = df_check[['barcode', 'well', 'workflow', 'filename_10x', 'filepath_10x']]
        df_check.to_csv(os.path.join(proj_folder, cell_line + '_' + str(dataset_id) + '_10x.csv'))

