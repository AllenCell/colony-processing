# preliminary analysis

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from variance_fov_colony.data.load_dataset import load_cell_meta_csv, load_passaging_info, load_image_date, load_imaging_mode

all_df = load_cell_meta_csv()
imaging_mode = load_imaging_mode()
imaging_date = load_image_date()
passaging_info = load_passaging_info()

all_df = all_df.merge(imaging_mode, on=['WellId'])
all_df = all_df.merge(imaging_date, on=['WellId'])
all_df = all_df.merge(passaging_info[['WellId', 'SourcePassage', 'flag_cell_population', 'PassageTotal', 'PassagePostThaw']], on=['WellId'])

plot_folder = r'\\allen\aics\microscopy\Calysta\test\variance_position\plots\confluency_all'

# check confluency before filtering all FOVs outside overview or touching boundary
df = all_df.loc[(all_df['meta_plate_bad_segmentation'] == 1)]
# short list of 10x images info only
df_plot = df[['structure_name', 'WellId', 'meta_confluency', 'WorkflowId', 'ImagingMode']].drop_duplicates()
df_plot = df_plot.loc[
    (df_plot['ImagingMode'] == 'Mode A')
    # (df_plot['structure_name'].isin(['H2B', 'NUP153', 'MYH10', 'RAB5A', 'SLC25A17']))
]
plt.figure()
sns.swarmplot(y='meta_confluency', x='structure_name', data=df_plot)
plt.ylim((0, 1))
plt.title('Confluency (bad_segmentation)')
plt.savefig(os.path.join(plot_folder, 'confluency_bad_seg.png'))
plt.show()

# filter after flags
df = all_df.loc[(all_df['meta_plate_bad_segmentation'] == 0) &
                (all_df['meta_fov_outside_overview'] == 0) &
                (all_df['meta_colony_touching_boundary'] == 0) &
                (all_df['flag_cell_population'] == 0)
                # & (all_df['structure_name'].isin(['H2B', 'NUP153', 'MYH10', 'RAB5A', 'SLC25A17']))
                ]

df = df[
    ['structure_name', 'FOVId', 'fov_path', 'dataset', 'BarCode',
     # 'expand_density_fov_area_2dmid',
     'InstrumentId', 'WorkflowId', 'ImagingMode', 'WellId', 'FOVImageDate',
     'PassagePostThaw', 'PassageTotal',
     'meta_fov_edgedist', 'meta_confluency', 'meta_colony_dist_center', 'meta_colony_area', 'meta_colony_count',
     'mem_shape_volume_lcc', 'mem_position_depth_lcc',
     ]
].drop_duplicates()


math_ops = []
# for some variables, set log and sqrt columns
for column in ['meta_colony_area']:
    df['log(' + column + ')'] = np.log(df[column])
    math_ops.append('log(' + column + ')')
    df['sqrt(' + column + ')'] = np.sqrt(df[column])
    math_ops.append('sqrt(' + column + ')')

## get general statistics of what is left per cell line# get counts per experimental setting
for struc in df['structure_name'].unique():
    print(struc)
    df_cell = df.loc[df['structure_name'] == struc]

    print('total: ' + str(len(df_cell)))

    # mode distribution
    for mode in np.sort(df_cell['ImagingMode'].unique()):
        print('mode ' + mode + ': ' + str(len(df_cell.loc[df_cell['ImagingMode'] == mode])))

    # instrument distribution
    for instrument in np.sort(df_cell['InstrumentId'].unique()):
        print(instrument + ': ' + str(len(df_cell.loc[df_cell['InstrumentId'] == instrument])))

    # pipeline distribution
    for workflow in np.sort(df_cell['WorkflowId'].unique()):
        print(workflow + ': ' + str(len(df_cell.loc[df_cell['WorkflowId'] == workflow])))

    # number of wells imaged from
    print('total number of wells: ' + str(len(df_cell['WellId'].unique())))
    #for well in np.sort(df_cell['WellId'].unique()):
    #    print(str(well) + ': ' + str(len(df_cell.loc[df_cell['WellId'] == well])))

    print('')


sns.set(font_scale=1)
sns.set_style("whitegrid", {'axes.grid' : False})
# Distribution for particular modes
for mode in np.sort(df_cell['ImagingMode'].unique()):
    df_plot = df.loc[df['ImagingMode'] == mode]
    # df_plot = df_plot.loc[(df_plot['meta_confluency'] >= 0.4) & (df_plot['meta_confluency'] <= 0.65)]

    df_plot['relative_dist_to_cent'] = df_plot['meta_fov_edgedist'] / df_plot['sqrt(meta_colony_area)']

    density_metrics = [# 'expand_density_cell_area_2dmid', 'expand_density_neigh_dist_median','expand_density_fov_area_2dmid',
                       'mem_shape_volume_lcc', 'mem_position_depth_lcc']
    colony_metrics = ['meta_fov_edgedist', 'meta_confluency', 'meta_colony_dist_center', 'meta_colony_area']

    alpha_settings = {'expand_density_cell_area_2dmid': 0.7,
                      'expand_density_neigh_dist_median': 0.7,
                      'expand_density_fov_area_2dmid': 0.5,
                      'mem_shape_volume_lcc': 0.5,
                      'mem_position_depth_lcc': 0.5}

    #plt.figure()
    #sns.violinplot(x='structure_name', y='')

    struc_date, imaging_order = get_imaging_order(df_plot, df_plot['structure_name'].unique(), order_mode='median')

    struc_month, order_month = get_imaging_order(df_plot, df_plot['structure_name'].unique(), order_mode='median_month')

    for index, row in df_plot.iterrows():
        df_plot.loc[index, 'ImagingMonth'] = struc_month[row['structure_name']]

    # # variation in distribution by imaging info
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    count = 0
    for row in range (0, 2):
        for col in range (0, 2):
            g = sns.boxplot(
                x='structure_name',
                y=colony_metrics[count],
                data=df_plot,
                ax=ax[row, col],
                order=imaging_order,
                hue='ImagingMonth',
                dodge=False,
                hue_order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                           'November', 'December']
                # palette={'March':"g", 'August':"b", 'June':"k", 'December':"r", 'May':"y",
                #         'July':"b", 'October':"b", 'January':"b", 'February':"b", 'September':"b",
                #         'November':"b", 'April':"b"},
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=90)
            count += 1
            if count != 3:
                g._remove_legend(g.get_legend())
    fig.savefig(os.path.join(plot_folder, 'mode_' + mode + '_colony_violin.png'))
    plt.show()

    # plot by date
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    count = 0
    for row in range (0, 2):
        for col in range (0, 2):
            g = sns.boxplot(
                x=df_plot['FOVImageDate'].dt.strftime('%b-%Y'),
                y=colony_metrics[count],
                data=df_plot, ax=ax[row, col],
                order=pd.to_datetime(df_plot['FOVImageDate'].dt.strftime('%b-%Y').unique()).sort_values().strftime('%b-%Y')
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=90)
            count += 1

    fig.savefig(os.path.join(plot_folder, 'mode_' + mode + '_colony_by_month_violin.png'))
    plt.show()

    # variation in distribution by density
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    count = 0
    for row in range (0, 2):
        for col in range (0, 2):
            if count < len(density_metrics):
                g = sns.boxplot(x='structure_name', y=density_metrics[count],
                                data=df_plot, ax=ax[row, col],
                                order=imaging_order)
                g.set_xticklabels(g.get_xticklabels(), rotation=90)
                count += 1
                if count > 1:
                    g._remove_legend(g.get_legend())

    fig.savefig(os.path.join(plot_folder, 'mode_' + mode + '_density_violin.png'))
    plt.show()

    all_colony_metrics = colony_metrics + ['relative_dist_to_cent'] + math_ops

    # by date
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    count = 0
    for row in range (0, 2):
        for col in range (0, 2):
            if count < len(density_metrics):
                g = sns.boxplot(x=df_plot['FOVImageDate'].dt.strftime('%b'), y=density_metrics[count],
                                data=df_plot, ax=ax[row, col],
                                order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                g.set_xticklabels(g.get_xticklabels(), rotation=90)
                count += 1
                if count > 1:
                    g._remove_legend(g.get_legend())

    fig.savefig(os.path.join(plot_folder, 'mode_' + mode + '_density_by_date_violin.png'))
    plt.show()

    # # scatter plots
    for density_metric in density_metrics:
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
        count = 0
        for row in range(0, 3):
            for col in range(0, 2):
                if count < len(all_colony_metrics):
                    g = sns.scatterplot(x=density_metric, y=all_colony_metrics[count], hue='structure_name',
                                        hue_order=imaging_order, data=df_plot,
                                        ax=ax[row, col], alpha=alpha_settings[density_metric], size=2)
                    count += 1
                    if count != 5:
                        g._remove_legend(g.get_legend())
        fig.savefig(os.path.join(plot_folder, 'mode_' + mode + '_' + density_metric + '.png'))
        plt.show()

    from matplotlib import animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1.0, metadata=dict(artist='Me'),) #bitrate=1800)

    plot_lim = {
        'meta_confluency': (0, 1),
        'meta_colony_area': (0, 500000),
        'meta_fov_edgedist': (0, 700),
        'log(meta_colony_area)': (6, 14),
        'mem_shape_volume_lcc': (0, 5000000),
        'mem_position_depth_lcc': (20, 180)
    }

    for x_axis in ['meta_confluency', 'meta_colony_area', 'meta_fov_edgedist', 'log(meta_colony_area)']:
        for y_axis in ['mem_shape_volume_lcc', 'mem_position_depth_lcc']:

            fig = plt.figure()
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.xlim(plot_lim[x_axis])
            plt.ylim(plot_lim[y_axis])

            # plt.legend(fontsize='xx-small')
            def animate(i):
                data = df_plot.loc[df_plot['structure_name'] == imaging_order[i]]
                # plt.figure()
                graph = sns.scatterplot(y=y_axis, x=x_axis, data=data,
                                        size=1, alpha=0.5)
                graph.set_title(imaging_order[i])
                #fig.title = imaging_order[i]
                graph._remove_legend(graph.get_legend())

            ani = animation.FuncAnimation(fig, animate, frames=len(imaging_order), interval=1000, repeat=False)
            ani.save(os.path.join(plot_folder, y_axis +'_vs_' + x_axis + '.mp4'), writer=writer)

    count = 0
    for colony_metric in all_colony_metrics:
        for passage_info in ['PassagePostThaw', 'PassageTotal']:
            plt.figure(figsize=(35, 10))
            sns.boxplot(
                x=passage_info,
                y=all_colony_metrics[count],
                data=df_plot,
                # ax=ax[row, col],
                # order=imaging_order,
                hue='structure_name',
                dodge=True
                # palette={'March':"g", 'August':"b", 'June':"k", 'December':"r", 'May':"y",
                #         'July':"b", 'October':"b", 'January':"b", 'February':"b", 'September':"b",
                #         'November':"b", 'April':"b"},
            )
            # g.set_xticklabels(g.get_xticklabels(), rotation=90)
            plt.legend(fontsize='xx-small')
            plt.show()
            # plt.savefig(os.path.join(plot_folder, 'mode_' + mode + '_' + colony_metric + ' vs ' + passage_info + '.png'))
        count += 1



    # binned by quartile
    # control for meta_fov_edge_dist and meta_colony_area, see if there are variations in density metric
    # bin info in colony info, see how much density changes
    bin_labels = ['d1', 'd2', 'd3', 'd4', 'd5']
    binned_dist, bin_names = pd.qcut(df_plot['meta_fov_edgedist'], q=5, labels=bin_labels, retbins=True)
    # calculate label values
    bin_dict = {}
    count = 0
    for bin in bin_labels:
        bin_dict.update({bin: int(bin_names[count])})
        count += 1
    binned_dist = binned_dist.replace(bin_dict)
    bin_dist_names = binned_dist.unique()
    bin_dist_names.sort()

    bin_labels = ['a1', 'a2', 'a3', 'a4', 'a5']
    binned_area, bin_names = pd.qcut(df_plot['meta_colony_area'], q=5, labels=bin_labels, retbins=True)
    bin_dict = {}
    count = 0
    for bin in bin_labels:
        bin_dict.update({bin: int(bin_names[count]/1000)})
        count += 1
    binned_area = binned_area.replace(bin_dict)
    bin_area_names = binned_area.unique()
    bin_area_names.sort()
    print('binned_area divided by 1000')

    # calculate label values
    df_plot = df_plot.merge(pd.DataFrame(binned_dist).rename(columns={'meta_fov_edgedist': 'binned_dist'}),
                            left_index=True, right_index=True)
    df_plot = df_plot.merge(pd.DataFrame(binned_area).rename(columns={'meta_colony_area': 'binned_area'}),
                            left_index=True, right_index=True)

    # show binned data
    for colony_metric in ['meta_fov_edgedist', 'meta_colony_area', 'sqrt(meta_colony_area)', 'log(meta_colony_area)',
                          'meta_confluency']:
        y_data = 'expand_density_fov_area_2dmid'
        fig = plt.figure()
        with sns.plotting_context(context="notebook", font_scale=1):
            grid = sns.FacetGrid(df_plot, col='binned_dist', row='binned_area', hue='structure_name', legend_out=True,
                                 row_order=bin_area_names, col_order=bin_dist_names,
                                 sharex=True, sharey=False)

            bp = grid.map(
                sns.scatterplot, y_data, colony_metric, alpha=alpha_settings[y_data]
            ).add_legend()

        plt.savefig(os.path.join(plot_folder, 'mode_' + mode + '_binned_density(fov)_vs_' + colony_metric + '.png'))
        plt.close()

    # report counts in binned regions
    count_per_bin = pd.DataFrame(df_plot.groupby(['binned_dist', 'binned_area', 'structure_name']).size().unstack(fill_value=0).stack(), columns=['count'])

    for index, count in count_per_bin.iterrows():
        d_bin, a_bin, structure_name = index
        count_per_bin.loc[index, 'binned_dist'] = d_bin
        count_per_bin.loc[index, 'binned_area'] = a_bin
        count_per_bin.loc[index, 'structure_name'] = structure_name

        total_count = len(df_plot.loc[df_plot['structure_name'] == structure_name])
        count_per_bin.loc[index, '% cells in bin'] = count['count'] * 100 / total_count

    fig = plt.figure()
    with sns.plotting_context(context="notebook", font_scale=1):
        grid = sns.FacetGrid(count_per_bin, col='binned_dist', row='binned_area', legend_out=True,
                             row_order=bin_area_names, col_order=bin_dist_names,
                             sharex=True, sharey=True)
        bp = grid.map(
            sns.barplot, 'structure_name', '% cells in bin'
        ).add_legend()

        for ax in grid.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(15)
    plt.savefig(os.path.join(plot_folder, 'mode_' + mode + '_binned_cell_counts.png'))
    plt.close()
    #
    # plot density profile in binned regions
    fig = plt.figure()
    with sns.plotting_context(context="notebook", font_scale=1):
        grid = sns.FacetGrid(df_plot, col='binned_dist', row='binned_area', legend_out=True,
                             row_order=bin_area_names, col_order=bin_dist_names,
                             sharex=False, sharey=True)
        bp = grid.map(
            sns.stripplot, 'structure_name', 'expand_density_fov_area_2dmid', s=2, alpha=alpha_settings['expand_density_fov_area_2dmid']
        ).add_legend()

        for ax in grid.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(15)

    plt.savefig(os.path.join(plot_folder, 'mode_' + mode + '_binned_density_fov_profile.png'))
    # plt.close()

    # scatter plot confluency vs colony area, number

    with sns.plotting_context(context="notebook", font_scale=1):
        plt.figure()
        sns.scatterplot(x='meta_confluency', y='meta_colony_area', data=df_plot, hue='structure_name')
        plt.show()

    with sns.plotting_context(context="notebook", font_scale=1):
        plt.figure()
        sns.scatterplot(x='meta_confluency', y='meta_colony_count', data=df_plot, hue='structure_name')
        plt.show()

    with sns.plotting_context(context="notebook", font_scale=1):
        plt.figure()
        sns.scatterplot(x='meta_colony_count', y='meta_colony_area', data=df_plot, hue='structure_name')
        plt.show()

    with sns.plotting_context(context="notebook", font_scale=1):
        plt.figure()
        sns.scatterplot(x='meta_confluency', y='meta_fov_edgedist', data=df_plot, hue='structure_name')
        plt.show()

    with sns.plotting_context(context="notebook", font_scale=1):
        plt.figure()
        sns.scatterplot(x='meta_confluency', y='expand_density_fov_area_2dmid', data=df_plot, hue='structure_name')
        plt.show()


    # 3D plot
    structure_color = {'H2B': 'blue',
                       'MYH10': 'yellow',
                       'NUP153': 'green',
                       'RAB5A': 'red',
                       'SLC25A17': 'purple'
    }
    from mpl_toolkits.mplot3d import axes3d
    # fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(df_plot['expand_density_fov_area_2dmid'], df_plot['meta_colony_area'], df_plot['meta_fov_edgedist'],
               c=df_plot['structure_name'].apply(lambda x: structure_color[x]), alpha=0.3, s=1)
    ax.set_xlabel('expand_density_fov_area_2dmid')
    ax.set_ylabel('meta_colony_area')
    ax.set_zlabel('meta_fov_edgedist')
    ax.legend()
    # plt.show()


# specific range, try to draw box
df_box = df_plot.loc[
    (df_plot['meta_colony_area'] >= 33000) &
    (df_plot['meta_colony_area'] <= 100000)  &
    (df_plot['meta_fov_edgedist'] >= 178) &
    (df_plot['meta_fov_edgedist'] <= 400)
]
for colony_metric in ['meta_colony_area', 'meta_confluency', 'meta_fov_edgedist']:
    plt.figure()
    sns.scatterplot(x=colony_metric, y='expand_density_fov_area_2dmid', data=df_box, hue='structure_name', alpha=0.2)
    plt.savefig(os.path.join(plot_folder, 'mode_A_draw_box_density_vs_' + colony_metric + '.png'))
plt.figure()
sns.swarmplot(y='expand_density_fov_area_2dmid', x='structure_name', data=df_box)
plt.savefig(os.path.join(plot_folder, 'mode_A_draw_box_density.png'))

# multivariate polynomial regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

output_df = pd.DataFrame()
for struc in ['MYH10', 'NUP153', 'H2B']:
    print(struc)
    df_train = df.loc[
        (df['ImagingMode'] == mode) &
        (df['structure_name'] == struc) &
        (df['meta_confluency'] >= 0.4) & (df['meta_confluency'] <= 0.6) &
        (df['meta_colony_area'] <= 100000)
        #(df['meta_colony_count'] <= 70) & (df_plot['meta_colony_count'] >= 30)
        ]
    # print(len(df_train))

    # set sample size for 600
    df_train = df_train.sample(n=65)

    X = df_train[['meta_confluency', 'meta_colony_area', 'meta_fov_edgedist']]
    y = df_train[['expand_density_fov_area_2dmid', 'structure_name', 'WellId']]

    scale = StandardScaler()
    scaled_X = scale.fit_transform(X)
    scaled_y = scale.fit_transform(y[['expand_density_fov_area_2dmid']])

    y['scaled_y'] = pd.DataFrame(scaled_y, index=y.index)

    for n in range (0, 50):
        # x_train, x_test, y_train, y_test = train_test_split(scaled_X, y, train_size=50, test_size=15)

        # sample 10 wells
        train_wells = np.random.choice(y['WellId'].unique(), 10, replace=False)
        test_wells = np.random.choice(y.loc[y['WellId'].isin(train_wells) == False, 'WellId'].unique(), 3, replace=False)

        y_train = y.loc[y['WellId'].isin(train_wells)]
        y_test = y.loc[y['WellId'].isin(test_wells)]
        x_train = scaled_X[y['WellId'].isin(train_wells)]
        x_test = scaled_X[y['WellId'].isin(test_wells)]
        print(len(y_train))

        for degree in range(1, 15, 1):
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(x_train, y_train[['scaled_y']])

            y_plot = model.predict(x_test)

            # print(str(degree) + ': ' + str(model.score(x_test, y_test[['scaled_y']])))
            y_test['pred'] = pd.DataFrame(y_plot, index=y_test.index)

            row = {}
            row['structure_name'] = struc
            row['degree'] = degree
            row['score'] = model.score(x_test, y_test[['scaled_y']])
            row['run'] = n
            output_df = output_df.append(row, ignore_index=True)

            plt.figure()
            sns.scatterplot(x='scaled_y', y='pred', data=y_test,
                            hue='structure_name')
            plt.title(struc + ' - test set ' + "degree %d" % degree +'; $R^2$: %.2f' % model.score(x_test, y_test[['scaled_y']]))
            plt.savefig(os.path.join(plot_folder, struc + '_test_deg' + str(degree) + '.png'))
            plt.show()

        # with cross validation
        cv_results = cross_validate(model, scaled_X, scaled_y, cv=10)
        median_r = np.median(cv_results['test_score'])
        print(str(degree) + ': ' + str(median_r))

# evaluate after all the runs
plt.figure()
sns.pointplot(x='degree', y='score', data=output_df, hue='structure_name')
plt.title('test set')
plt.ylim((-5, 1))
plt.show()


# bokeh draw box, selection and changes in points across variables
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, tools, CategoricalColorMapper
from bokeh import palettes
from bokeh.transform import jitter
from bokeh.layouts import gridplot
from bokeh.models import Circle
# interactive plot for cell volume and nuc volume
TOOLS = "box_select, lasso_select, help, box_zoom, pan, wheel_zoom, reset"

# hover = tools.HoverTool()
# base_list = [('file_name', '@file_name'),
#              ('cell_num', '@cell_object_number')]

# hover.tooltips = base_list

df_plot = df.loc[df['expand_labkey_imode'] == mode]
colors = palettes.Category10[len(df_plot['structure_name'].unique())]
colormap = {}
count = 0
for i in df_plot['structure_name'].unique():
    colormap.update({i: colors[count]})
    count += 1
colors = [colormap[x] for x in df_plot.structure_name]

source = ColumnDataSource(df_plot)
from bokeh.palettes import d3
palette = d3['Category10'][len(df['structure_name'].unique())]
color_map = CategoricalColorMapper(factors=df_plot['structure_name'].unique(),
                                   palette=palette)

s1 = figure(tools=TOOLS, title='Confluency vs colony area', plot_width=800, plot_height=300)
r1 = s1.circle(x='meta_colony_area', y='meta_confluency',
           color={'field': 'structure_name', 'transform': color_map},
           source=source,
           size=2, alpha=0.2)
s1.xaxis.axis_label = "meta_colony_area"
s1.yaxis.axis_label = "meta_confluency"
r1.selection_glyph = Circle(fill_color="red", line_color=None)
r1.nonselection_glyph = Circle(fill_color="black", line_color=None)

s2 = figure(tools=TOOLS, title='Confluency vs colony count', plot_width=800, plot_height=300)
r2 = s2.circle(x='meta_colony_count', y='meta_confluency',
           color={'field': 'structure_name', 'transform': color_map},
           source=source,
           size=2, alpha=0.2)
s2.xaxis.axis_label = "meta_colony_count"
s2.yaxis.axis_label = "meta_confluency"
r2.selection_glyph = Circle(fill_color="red", line_color=None)
r2.nonselection_glyph = Circle(fill_color="black", line_color=None)

s3 = figure(tools=TOOLS, title='Colony area vs colony count', plot_width=800, plot_height=300)
r3 = s3.circle(x='meta_colony_area', y='meta_colony_count',
           color={'field': 'structure_name', 'transform': color_map},
           source=source,
           size=2, alpha=0.2)
s3.xaxis.axis_label = "meta_colony_count"
s3.yaxis.axis_label = "meta_colony_area"
r3.selection_glyph = Circle(fill_color="red", line_color=None)
r3.nonselection_glyph = Circle(fill_color="black", line_color=None)

s4 = figure(tools=TOOLS, title='Colony area vs density', plot_width=800, plot_height=300)
s4.xaxis.axis_label = "meta_colony_area"
s4.yaxis.axis_label = "expand_density_fov_area_2dmid"
r4 = s4.circle(x='meta_colony_count', y='expand_density_fov_area_2dmid',
           color={'field': 'structure_name', 'transform': color_map},
           source=source,
           size=2, alpha=0.2)
r4.selection_glyph = Circle(fill_color="red", line_color=None)
r4.nonselection_glyph = Circle(fill_color="black", line_color=None)

s5 = figure(tools=TOOLS, title='Colony area vs edge dist', plot_width=800, plot_height=300)
s5.xaxis.axis_label = "meta_colony_area"
s5.yaxis.axis_label = "meta_fov_edgedist"
r5 = s5.circle(x='meta_colony_count', y='meta_fov_edgedist',
           color={'field': 'structure_name', 'transform': color_map},
           source=source,
           size=2, alpha=0.2)
r5.selection_glyph = Circle(fill_color="red", line_color=None)
r5.nonselection_glyph = Circle(fill_color="black", line_color=None)

s6 = figure(tools=TOOLS, title='edge dist vs density', plot_width=800, plot_height=300)
s6.xaxis.axis_label = "meta_fov_edgedist"
s6.yaxis.axis_label = "expand_density_fov_area_2dmid"
r6 = s6.circle(x='meta_fov_edgedist', y='expand_density_fov_area_2dmid',
           color={'field': 'structure_name', 'transform': color_map},
           source=source,
           size=2, alpha=0.2)
r6.selection_glyph = Circle(fill_color="red", line_color=None)
r6.nonselection_glyph = Circle(fill_color="black", line_color=None)

# jitter for density across cell lines
from bokeh.transform import jitter
s7 = figure(tools=TOOLS, title='density profile', plot_width=800, plot_height=300, x_range=df_plot['structure_name'].unique())
r7 = s7.circle(x=jitter('structure_name', width=0.6, range=s7.x_range), y='expand_density_fov_area_2dmid', source=source,
               size=2, color={'field': 'structure_name', 'transform': color_map}
          )
r7.selection_glyph = Circle(fill_color="red", line_color=None)
r7.nonselection_glyph = Circle(fill_color="black", line_color=None)

p = gridplot([[s1, s2], [s3, s5], [s4, s6], [s7]])
show(p)

from bokeh.plotting import figure, output_file, save
output_file("test.html")
save(p)

structure_list = ['H2B', 'ATP2A2']
test = df.loc[df['structure_name'].isin(structure_list)]

def get_imaging_order(df, structure_list, order_mode='median'):
    structure_date = {}
    for structure in structure_list:
        if order_mode == 'median':
            struc_time = df.loc[df['structure_name'] == structure, 'FOVImageDate'].quantile(0.5)
        elif order_mode == 'min':
            struc_time = df.loc[df['structure_name'] == structure, 'FOVImageDate'].quantile(0)
        elif order_mode == 'min':
            struc_time = df.loc[df['structure_name'] == structure, 'FOVImageDate'].quantile(1)
        elif order_mode == 'median_month':
            struc_time = df.loc[df['structure_name'] == structure, 'FOVImageDate'].quantile(0.5).month_name()
        elif order_mode == 'min_month':
            struc_time = df.loc[df['structure_name'] == structure, 'FOVImageDate'].quantile(0).month_name()
        elif order_mode == 'max_month':
            struc_time = df.loc[df['structure_name'] == structure, 'FOVImageDate'].quantile(1).month_name()

        structure_date.update({structure: struc_time})
    sort_date = sorted(structure_date.items(), key=lambda x: x[1], reverse=False)
    sorted_structure_list = []
    for struc, time in sort_date:
        sorted_structure_list.append(struc)

    return structure_date, sorted_structure_list

