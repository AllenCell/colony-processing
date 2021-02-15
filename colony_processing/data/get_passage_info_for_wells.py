# Get passaging info for wells

import pandas as pd
from lkaccess import LabKey
from variance_fov_colony.data.load_dataset import load_shape_space_csv

lk = LabKey(host='aics.corp.alleninstitute.org')

all_df = load_shape_space_csv()
all_well_id = all_df['WellId'].unique()

query_filter = ''
for well in all_well_id:
    query_filter += ';' + str(well)

query_wells = lk.select_rows_as_list(
    schema_name='microscopy',
    query_name='WellCellPopulationJunction',
    filter_array = [
        ('WellId/WellId', query_filter[1:], 'in')
    ],
    columns=[
        'WellId/PlateId/BarCode',
        'WellId/WellName/Name',
        'ParentVialId',
        'ParentVialId/Name',
        'ParentCellPopulationId',
        'ParentWellId',
        'CellPopulationStateId',
        'SeedingDensity',
        'Viability',
        'WellId',
        'CellPopulationId',
        'CellPopulationId/Passage',
    ]
)
df = pd.DataFrame(query_wells)

# fix flagged cell population (cannot fix 1288, 6248 for lost info)

# 1. if 96 well is seeded from source directly, set parent well as itself
df.loc[df['ParentVialId/Name'].isnull() == False, 'ParentWellId'] = df.loc[df['ParentVialId/Name'].isnull() == False, 'WellId']

# 2. Correct for empty parent wells, add back to dataframe
fix_parent_wells = {
    '3500001829': [120356, 5415],
    '3500001827': [120356, 5415],
    '3500000933': [119215, 5214],
    '3500000935': [119215, 5214],
    '3500001464': [119874, 5351],
    '3500001830': [120360, 5389],
    '3500001831': [120360, 5389],
    '3500001937': [120502, 5441],
    '3500001941': [120502, 5441],
    '3500001945': [120502, 5441],
    '3500002033': [134770, 6672],
    '3500002035': [134770, 6672],
    '3500002036': [134770, 6672],
    '3500002038': [134770, 6672],
}

for index, row in df.iterrows():
    if row['WellId/PlateId/BarCode'] in fix_parent_wells.keys():
        if row['ParentCellPopulationId'] == fix_parent_wells[row['WellId/PlateId/BarCode']][1]:
            df.loc[index, 'ParentWellId'] = fix_parent_wells[row['WellId/PlateId/BarCode']][0]

# 2. Parent well ID linked to a plate, but the plate does not have vial info
fix_parent_well_vial = {
    134740.: 'AI0057:00009276',
    134742.: 'AI0057:00009276',
    134752.: 'AI0057:00009276',
    134796.: 'AI0058:00009816',
    134816.: 'AI0058:00009816',
    134770.: 'AI0058:00009816'
}

for parent_well_id, vial_id in fix_parent_well_vial.items():
    df.loc[df['ParentWellId'] == parent_well_id, 'ParentVialId/Name'] = vial_id

# 3. Fix wrong cell passage number
fix_passage = {
    '3500001191': 38,
    '3500001198': 38,
    '3500001205': 39,
    '3500001217': 40,
    '3500001218': 40,
    '3500001152': 35,
    '3500001153': 35,
    '3500001157': 36,
    '3500001175': 37,
    '3500001179': 37
}

for barcode, passage in fix_passage.items():
    df.loc[df['WellId/PlateId/BarCode'] == barcode, 'CellPopulationId/Passage'] = passage

def get_parent_cell_population_id(parent_well, current_passage):
    parent_cell_population_id = None
    parent_vial = None
    flag_lineage = False
    error_lineage = []


    relative_passage = 0
    while (parent_cell_population_id is None) & (parent_vial is None):
        relative_passage += 1
        query_wells = lk.select_rows_as_list(
            schema_name='microscopy',
            query_name='WellCellPopulationJunction',
            filter_array=[
                ('WellId/WellId', int(parent_well), 'eq')
            ],
            columns=[
                'WellId/PlateId/BarCode',
                'WellId/WellName/Name',
                'ParentVialId',
                'ParentVialId/Name',
                'ParentCellPopulationId',
                'ParentWellId',
                'CellPopulationStateId',
                'SeedingDensity',
                'Viability',
                'WellId',
                'CellPopulationId',
                'CellPopulationId/Passage'
            ]
        )

        df_parent = pd.DataFrame(query_wells)

        for fix_cell in [1288]:
            if df_parent.loc[0, 'CellPopulationId'] == fix_cell:
                parent_cell_population_id = fix_cell

        if len(df_parent) == 1:
            if (df_parent['ParentVialId'].isnull().loc[0]) & (df_parent['ParentWellId'].isnull().loc[0] == False):
                parent_well = df_parent.loc[0, 'ParentWellId']
                # print(parent_well)
            else:
                parent_vial = df_parent.loc[0, 'ParentVialId/Name']
                parent_cell_population_id = df_parent.loc[0, 'ParentCellPopulationId']
                # print(parent_cell_population_id)
            parent_passage = df_parent.loc[0, 'CellPopulationId/Passage']
            print(parent_passage, current_passage)
            if (current_passage - parent_passage) != 1:
                flag_lineage = True
                error_lineage.append(parent_well)
            current_passage = parent_passage

    return parent_cell_population_id, parent_vial, relative_passage, error_lineage, flag_lineage


def get_source_passage_from_cell(parent_cell_population_id):
    passage = None
    query_cells = lk.select_rows_as_list(
        schema_name='celllines',
        query_name='CellPopulation',
        filter_array=[
            ('CellPopulationId', int(parent_cell_population_id), 'eq')
        ]
    )
    df_cell = pd.DataFrame(query_cells)

    if len(df_cell) == 1:
        passage = df_cell.loc[0, 'Passage']

    return passage


def get_source_passage_from_vial(vial_id):
    passage = None

    query_vial = lk.select_rows_as_list(
        schema_name='celllines',
        query_name='Vial',
        filter_array=[
            ('Name', vial_id, 'eq')
        ],
        columns=[
            'VialId',
            'Name',
            'CellBatchId/CellPopulationId/Passage'
        ]
    )

    df_vial = pd.DataFrame(query_vial)
    if len(df_vial) == 1:
        passage = df_vial.loc[0, 'CellBatchId/CellPopulationId/Passage']

    return passage


df_parent = pd.DataFrame()
count = 0
for parent_well in df['ParentWellId'].dropna().unique():
    # print('processing ' + str(count) + ' out of ' + str(len(df['ParentWellId'].dropna().unique())))
    cell_id = None
    vial_id = None
    passage = None
    later = None

    try:
        if parent_well in fix_parent_well_vial.keys():
            vial_id = fix_parent_well_vial[parent_well]
        elif (cell_id is None) & (vial_id is None):
            barcode = int(df.loc[df['ParentWellId'] == parent_well, 'WellId/PlateId/BarCode'].unique()[0])
            if barcode in [9999999104, 9999999107, 3500000968, 3500000964, 9999999185, 9999999184]:
                current_passage = df.loc[df['ParentWellId'] == parent_well, 'CellPopulationId/Passage'].unique()[0]
                fix_96_passage = False
            elif (barcode < 3500002359) | (barcode > 9999990000):
                current_passage = df.loc[df['ParentWellId'] == parent_well, 'CellPopulationId/Passage'].unique()[0] + 1
                fix_96_passage = True
            else:
                current_passage = df.loc[df['ParentWellId'] == parent_well, 'CellPopulationId/Passage'].unique()[0]
                fix_96_passage = False
            cell_id, vial_id, calc_relative_passage, error_lineage, flag_lineage = get_parent_cell_population_id(parent_well,
                                                                                                                 current_passage)
    except:
        cell_id = None
        vial_id = None
        pass

    if vial_id is not None:
        passage = get_source_passage_from_vial(vial_id)

    if cell_id is not None:
        passage = get_source_passage_from_cell(cell_id)

    row = {}
    row['ParentWellId'] = parent_well
    row['ParentCellId'] = cell_id
    row['VialId'] = vial_id
    row['SourcePassage'] = passage
    row['CalculatedPassage'] = calc_relative_passage
    row['FlagLineage'] = flag_lineage
    row['ErrorLineage'] = error_lineage
    row['Fix96Passage'] = fix_96_passage
    df_parent = df_parent.append(row, ignore_index=True)
    count += 1

new_df = df.merge(df_parent, on='ParentWellId', how='left')
new_df['flag_cell_population'] = False
new_df.loc[new_df['SourcePassage'].isnull(), 'flag_cell_population'] = True

new_df.loc[new_df['Fix96Passage'] == True, 'CellPopulationId/Passage'] = new_df['CellPopulationId/Passage'] + 1

fix_error = {
    119486: -8,
    119487: 8,
    119833: -1,
    120034: -1,
}

for index, row in new_df.loc[new_df['FlagLineage'] == True].iterrows():
    for error in row['ErrorLineage']:
        if error in fix_error.keys():
            print(error)
            new_df.loc[index, 'FlagLineage'] = False
            new_df.loc[index, 'CellPopulationId/Passage'] = row['CellPopulationId/Passage'] + fix_error[error]

new_df['PassagePostThaw'] = new_df['CellPopulationId/Passage'] - new_df['SourcePassage']

new_df.loc[new_df['FlagLineage'] == True, 'flag_cell_population'] = new_df['FlagLineage']

new_df = new_df.rename(columns={
    'WellId/PlateId/BarCode': 'BarCode',
    'WellId/WellName/Name': 'WellName',
    'VialId': 'SourceVialId',
    'CellPopulationId/Passage': 'PassageTotal'
}
)

new_df = new_df[[
    'WellId',
    'BarCode',
    'WellName',
    'SourceVialId',
    'SourcePassage',
    'PassageTotal',
    'PassagePostThaw',
    'CalculatedPassage',
    'flag_cell_population',
]]

new_df.to_csv(r'\\allen\aics\microscopy\Data\fov_in_colony\passage_info_with_calc.csv')

