from ..default import *
from .. import structure
from . import figure_unit

import matplotlib.gridspec as gridspec

def plot(self, plot_params=None, save=False, return_fig=False):
    'setting'
    if plot_params:
        for k in self.default_plot_params:
            if k not in plot_params:
                plot_params[k] = self.default_plot_params[k]
    else:
        plot_params = self.default_plot_params
    # if 'color' in plot_params:
    #     if isinstance(plot_params['color'], str):
    #         plot_params['color'] = sns.color_palette(plot_params['color'])
    #     else:
    #         plot_params['color'] = plot_params['color']
    if 'style' in plot_params:
        sns.set_style(plot_params['style']) # "ticks" "white" "dark" 'darkgrid'(default) 'whitegrid'

    if 'x_len' in plot_params:
        fig_x = plot_params['x_len']
    else:
        fig_x = 8

    'preparing the canvas'
    fig = plt.figure(figsize=(fig_x,5))
    if plot_params['plot_type'][0] == 'direct':
        ax = fig.add_subplot(111)
        select_subplot_type(plot_params['plot_type'][1], ax, self.data, self.annotation, plot_params)
    elif plot_params['plot_type'][0] == 'matrix':
        matrix_plot(self.data, plot_params['x_title'], plot_params['y_title'], 
                    plot_params, self.annotation, fig)
    elif plot_params['plot_type'][0] == 'float':
        float_plot(self.data, plot_params['xy_locs'], 
                    plot_params, self.annotation, fig)
    else:
        raise Exception(f'Unsupported plot_type {plot_params["plot_type"][0]}')  

    'output'
    plt.show()
    if save:
        title = f'{title}.png'
        if isinstance(save, str):
            title = save
        fig.savefig(title, transparent=True)

    'reset'
    sns.despine()
    sns.set()
    
    if return_fig:
        return fig

structure.Analyzed_data.plot = plot

def select_subplot_type(subplot_type, ax, data, annotation, plot_params):
    if subplot_type == 'waveform':
        figure_unit.plot_waveform(data, plot_params, ax)
        if 'compare' in plot_params and plot_params['compare']==True:
            figure_unit.significant(annotation, plot_params['win'], plot_params['sig_limit'], ax)
    elif subplot_type == 'spectrum':
        figure_unit.plot_spectrum(data, plot_params, ax)
        if 'compare' in plot_params and plot_params['compare']==True:
            figure_unit.significant(annotation, plot_params['win'], plot_params['sig_limit'], ax)
    elif subplot_type == 'topograph':
        figure_unit.plot_topograph(data, annotation, plot_params, ax)
    elif subplot_type == 'heatmap':
        figure_unit.plot_heatmap(data, plot_params, ax)
        if 'compare' in plot_params and plot_params['compare']==True:
            figure_unit.heatmap_significant(annotation, plot_params['sig_limit'], ax)
    else:
        raise Exception(f'Unsupported subplot_type "{subplot_type}"')


def float_plot(data, positions, plot_params, annotation=None, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(8, 5))

    if positions is 'channels':
        positions = io.load_topolocs('standard-10-5-cap385', None)

    sns.set_style("white")

    data_cells = dict((k[2:],v) for k,v in data.groupby(level='channel_group'))

    if isinstance(annotation, pd.DataFrame):
        annotation_cells = dict((k[2:],v) for k,v in annotation.groupby(level='channel_group'))
    
    axs = []
    for idx,(data_name,data) in enumerate(data_cells.items()):
        data.name = data_name
        sys.stdout.write(' ' * 30 + '\r')
        sys.stdout.flush()
        sys.stdout.write('%d/%d\r' %(idx+1,len(data_cells)))
        sys.stdout.flush()

        axs.append(fig.add_axes([idx,0,0,0]))

        if isinstance(annotation, pd.DataFrame):
            select_subplot_type(plot_params['plot_type'][1], axs[idx], data, annotation_cells[data_name], plot_params)
        else:
            select_subplot_type(plot_params['plot_type'][1], axs[idx], data, None, plot_params)
        axs[idx].set_position([positions[data_name][0], positions[data_name][1] ,0.3,0.3])
        axs[idx].axis('off')

        axs[idx].legend(bbox_to_anchor=(1.01, 1))

    sns.set() # switch to seaborn defaults

def matrix_plot(data, x_axis, y_axis, plot_params, annotation=None, fig=None):
    def pre(target):
        target = target.stack('time')

        for level_name in target.index.names:
            if '_group' in level_name:
                old_values_in_level = target.index.levels[target.index.names.index(level_name)]
                target.index = target.index.set_levels([' '.join(i.split(' ')[1:]) for i in old_values_in_level],level=level_name)
    
        'add "ms" to time axis'
        old_values_in_level = target.index.levels[target.index.names.index('time')]
        target.index = target.index.set_levels([f'{i}ms' for i in old_values_in_level],level='time')
        return target

    if fig is None:
        fig = plt.figure(figsize=(8, 5))

    if not 'title' in plot_params:
        plot_params['title'] = ''

    if plot_params['plot_type'][1] == 'topograph':
        if not 'zlim' in plot_params:
            plot_params['zlim'] = (data.min().min(), data.max().max())
        if not 'cbar_title' in plot_params:
            plot_params['cbar_title'] = ''
        if not 'color' in plot_params:
            plot_params['color'] = plt.cm.jet
        if not 'chan_locs' in plot_params:
            from .. import io
            plot_params['chan_locs'] = io.load_topolocs('standard-10-5-cap385', None)

    data = pre(data)
    data_cells = dict((k, v) for k, v in data.groupby(level=[y_axis, x_axis]))
    if not annotation is None:
        annotation = pre(annotation)
        annotation_cells = dict((k, v) for k, v in annotation.groupby(level=[y_axis, x_axis]))
    else:
        annotation_cells = dict((k, None) for k in data_cells.keys())

    x_axis_values = list(OrderedDict.fromkeys(data.index.get_level_values(x_axis))) # remove duplicates from a list in whilst preserving order
    y_axis_values = list(OrderedDict.fromkeys(data.index.get_level_values(y_axis))) # remove duplicates from a list in whilst preserving order

    col_N,row_N = len(x_axis_values), len(y_axis_values)

    sns.set_style("white")
    fig.set_figwidth(15)

    W,H = 1,1

    wspace = 0.1
    hspace = 0.1
    has_title = False
    has_colorbar = False

    height_ratios = [0,0.3,row_N,1]
    width_ratios = [0.7,col_N,0.1]
    fig.set_figwidth(sum(width_ratios)*1.7)
    fig.set_figheight(sum(height_ratios)*1.7)

    if has_title:
        height_ratios[0] = 0.5

    inner_grids = gridspec.GridSpec(4, 3, wspace=wspace, hspace=hspace, height_ratios=height_ratios, width_ratios=width_ratios)
    
    'title'
    ax = plt.Subplot(fig, inner_grids[1])
    ax.text(0.5, 0.5, plot_params['title'], va="center", ha="center")
    ax.axis('off')
    fig.add_subplot(ax)

    'x_axis'
    inner_grids_for_x_axis = gridspec.GridSpecFromSubplotSpec(1, col_N, inner_grids[4], wspace=wspace, hspace=hspace)
    for i in range(col_N):
        ax = plt.Subplot(fig, inner_grids_for_x_axis[i])
        ax.text(0.5, 0.5, x_axis_values[i], va="center", ha="center")
        ax.axis('off')
        fig.add_subplot(ax)

    'y_axis'
    inner_grids_for_y_axis = gridspec.GridSpecFromSubplotSpec(row_N, 1, inner_grids[6], wspace=wspace, hspace=hspace)
    for i in range(row_N):
        ax = plt.Subplot(fig, inner_grids_for_y_axis[i])
        ax.text(0.5, 0.5, y_axis_values[i], va="center", ha="center")
        ax.axis('off')
        fig.add_subplot(ax)

    'data'
    inner_grids_for_data = gridspec.GridSpecFromSubplotSpec(row_N, col_N, inner_grids[7])
    
    for idx,data_name in enumerate(itertools.product(y_axis_values,x_axis_values)):
        ax = plt.Subplot(fig, inner_grids_for_data[idx])
        fig.add_subplot(ax)
        select_subplot_type(plot_params['plot_type'][1], ax,
                            data_cells[data_name], annotation_cells[data_name], plot_params)

    'color_bar'
    inner_grids_for_cbar = gridspec.GridSpecFromSubplotSpec(row_N, 1, inner_grids[8], wspace=wspace, hspace=hspace)
    ax = plt.Subplot(fig, inner_grids_for_cbar[row_N-1])
    cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=plot_params['color'] ,
                                    norm=matplotlib.colors.Normalize(vmin=plot_params['zlim'][0], vmax=plot_params['zlim'][1]),
                                    orientation='vertical')
    if 'cbar_title' in plot_params:
        ax.set_title(plot_params['cbar_title'])

    fig.add_subplot(ax)       

    sns.set() # switch to seaborn defaults
