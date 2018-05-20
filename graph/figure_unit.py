from ..default import *

import math
from pylab import get_cmap
from matplotlib.patches import Rectangle

def cmap_discretize(cmap, N):
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0.1, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
    
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def refine_axis(title, xticklabels, plot_params, ax):
    xticks = ax.get_xticks()
    #  or len(xticklabels) > 40
    if len(xticks) >= 40:
        if isinstance(xticklabels, pd.MultiIndex):
            xticklabels = xticklabels.get_level_values('time')

        if xticklabels[-1] - xticklabels[0] < 500:
            step = 50
        else:
            step = 100
        step_num = len(np.arange(xticklabels[0], xticklabels[-1]+5, step))

        xticks = ax.get_xticks()
        xticks = np.linspace(xticks[0],xticks[-1],step_num)
        xticklabels = np.linspace(xticklabels[0],xticklabels[-1],step_num,dtype=int)
        xticklabels_min = xticklabels[0]
        xticklabels_fine = [i//step*step for i in xticklabels]
        xticklabels[0] += 1  # prevent label-xticklabels_min is zero
        xticks_fine = [(label_fine-xticklabels_min)/(label-xticklabels_min)*tick for tick,label,label_fine in zip(xticks,xticklabels,xticklabels_fine)]
        xticks_fine[0] = 0  # prevent xtick_min is less than zero
        
        # padding the last x number in case it lost 
        if xticks[-1] - xticks_fine[-1] > (xticks_fine[-1] - xticks_fine[-2])/2:
            xticks_fine += [xticks_fine[-1] + xticks_fine[-1] - xticks_fine[-2]]
            xticklabels_fine += [xticklabels_fine[-1] + xticklabels_fine[-1] - xticklabels_fine[-2]]
            
        ax.set_xticks(xticks_fine)
        ax.set_xticklabels(xticklabels_fine,rotation=45,fontname='Consolas',size='large')
    else:
        if ax.get_xticklabels()[0].get_text() != ax.get_xticklabels()[1].get_text():
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45,fontname='Consolas',size='large')

    if 'x_title' in plot_params:
        ax.set_xlabel(plot_params['x_title'])
    if 'y_title' in plot_params:
        ax.set_ylabel(plot_params['y_title'])

    if 'title' in plot_params:
        ax.set_title(plot_params['title'])
    else:
        ax.set_title(title)

    if 'ylim' in plot_params:
        ax.set_ylim(plot_params['ylim']) # limitation on y-axis


def heatmap_significant(pv_data, sig_limit=0.05, ax=None):
    from matplotlib.patches import Rectangle

    significance = [(col_index, row_index[0])
                    for row_index, row in pv_data.iterrows() for col_index, cell in row.iteritems()
                    if cell < sig_limit]
                    
    for x, y in significance:
        ax.add_patch(Rectangle((pv_data.columns.get_level_values('time').max() - x, 
                                pv_data.index.get_level_values('freq').max() - y), 
                                2, 1, alpha=0.5))


def significant(pv_data, win, sig_limit=0.05, ax=None):
    win = int(win[:-2])
    xmin, xmax, ymin, ymax = ax.axis()

    sig_plot_list = np.zeros(int(xmax-xmin))

    for limit,alpha in zip([sig_limit,0.05,0.01,0.001],[0.1,0.2,0.3,0.4]):
        for tp,pv_df in pv_data.items():
            # ugly code
            if isinstance(tp, tuple):
                tp = tp[1]

            pv = pv_df.values[0]
            if pv and pv<limit<=sig_limit:
                sig_plot_list[int(tp)-win//2:int(tp)+win//2] = alpha

    for tp,alpha in enumerate(sig_plot_list):
        if alpha>0:
            ax.axvspan(tp,tp+1, facecolor='0', alpha=alpha, edgecolor='none')

    cbar_ax = ax.get_figure().add_axes(
        [0.95, 0.4, 0.01, 0.2])  # [left, bottom, width, height]

    cmaplist = [(.2, .2, .2, 1.0), (.3, .3, .3, 1.0), (.4, .4, .4, 1.0)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, 3)

    # define the bins and normalize
    bounds = np.linspace(0, 3, 4)
    norm = matplotlib.colors.BoundaryNorm(bounds, 3)
    ticks = [(bounds[i]+bounds[i+1])/2 for i in range(len(bounds)-1)]

    cb = matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, spacing='proportional', ticks=ticks, boundaries=bounds, format='%1i')

    cbar_ax.set_title('pvalue')
    cbar_ax.set_yticklabels(['<0.05', '<0.01', '<0.001'])

def plot_waveform(data, plot_params={'err_style':'ci_band', 'color':"Set1"}, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    flat_data = pd.DataFrame(data.stack('time'))
    # flat_data.reset_index(level=['condition_group','time'], inplace=True)
    flat_data.reset_index(level=flat_data.index.names, inplace=True)
    
    
    if 'condition_group' in flat_data.columns:
        flat_data['condition_group'] = flat_data['condition_group'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    if 'channel_group' in flat_data.columns:
        flat_data['channel_group'] = flat_data['channel_group'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    
    if 'channel_group' in flat_data.columns and len(flat_data['channel_group'].unique())>1:
        group = "channel_group"
    else:
        group = "condition_group"

    if 'subject' not in flat_data.columns:
        flat_data['subject'] = 0

    if len(flat_data[group].unique()) < 10:
        legend = True
    else:
        legend = False

    if 'err_style' not in plot_params:
        plot_params['err_style'] = None

    sns.tsplot(time="time", value=0, unit='subject', condition=group, data=flat_data, ax=ax, 
        err_style=plot_params['err_style'], legend=legend, color=plot_params['color'])

    xticklabels = flat_data.columns

    refine_axis(data.name, xticklabels, plot_params, ax)

def plot_spectrum(data, plot_params={'err_style':'ci_band', 'color':"Set1"}, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    flat_data = data.stack('frequency')
    flat_data = pd.DataFrame(flat_data, columns=['power'])
    flat_data.reset_index(level=['condition_group','channel_group','subject','frequency'], inplace=True)
    flat_data['condition_group'] = flat_data['condition_group'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    flat_data['channel_group'] = flat_data['channel_group'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    
    if len(flat_data['channel_group'].unique())>1:
        group = "channel_group"
        legend = False
    else:
        group = "condition_group"
        legend = True

    sns.tsplot(time="frequency", value='power', unit="subject", condition=group,
        data=flat_data, ax=ax, 
        err_style=plot_params['err_style'], legend=legend, color=plot_params['color'])

    xticklabels = flat_data.columns

    refine_axis(data.name, xticklabels, plot_params, ax)


def plot_heatmap(data, plot_params={'grid':True,'color': sns.cubehelix_palette(light=1, as_cmap=True)}, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    cbar_ax = ax.get_figure().add_axes([0.95,0.4,0.01,0.2]) # [left, bottom, width, height]
    name = data.name
    if 're_assign' in plot_params:
        # if 'cbar_values' not in plot_params:
        #     plot_params['cbar_values'] = plot_params['re_assign'][0]
        def re_assign(v):
            scale = plot_params['re_assign']
            for idx, new_v in enumerate(scale[1]):
                if (v >= scale[0][idx]) and (v < scale[0][idx+1]):
                    return new_v
        data[data==1] = 1-1e-6
        data = data.applymap(lambda v: re_assign(v))
        
    data.name = name

    if data.index.name and '_group' in data.index.name:
        data.index = [' '.join(i.split(' ')[1:]) for i in data.index]

    if 'cbar_values' in plot_params:
        cmap=cmap_discretize(plot_params['color'],len(plot_params['cbar_values']))
        sns.heatmap(data,ax=ax,cbar_ax=cbar_ax,cmap=cmap)
        cb_yticks = cbar_ax.get_yticks()
        cbar_ax.yaxis.set_ticks([(cb_yticks[i]+cb_yticks[i+1])/2*1.5 for i in range(len(cb_yticks)-1)])
        cbar_ax.set_yticklabels(plot_params['cbar_values'])
    else:
        sns.heatmap(data,ax=ax,cbar_ax=cbar_ax,cmap=plot_params['color'])

    xticklabels = data.columns

    refine_axis(data.name, xticklabels, plot_params, ax)

    if len(xticklabels)<40:
        ax.set_aspect(1) # ratio between y_unit and x_unit
    else:
        ax.set_aspect(len(xticklabels)/40)

    ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontname='Consolas')

    cbar_ax.set_title(plot_params['cbar_title'])

    if 'grid' in plot_params and plot_params['grid']:
        for i in ax.get_xticks():
            ax.axvline(i, c="w",linewidth=1,linestyle='dotted')
        for i in range(len(data.index)):
            ax.axhline(i, c="w",linewidth=2)

def get_topograph(data, locations, channels, N):
    locs = np.array([locations[chan] for chan in channels])

    x_span = locs[:,1].max()-locs[:,1].min()
    y_span = locs[:,0].max()-locs[:,0].min()
    xi = np.linspace(0, x_span, N)[None,:]
    yi = np.linspace(0, y_span, N)[:,None]

    locs = [(x-locs[:,1].min(),y-locs[:,0].min()) for y,x in locs]
    locs.extend([(0,0), (0,y_span), (x_span,0), (x_span,y_span)])

    return scipy.interpolate.griddata(locs, np.append(data, [0,0,0,0]), (xi, yi), method='cubic')

def plot_topograph(amp_data, pvalue_data, plot_params={'color': plt.cm.jet}, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    amp_data = pd.DataFrame(amp_data,columns = ['Amp'])
    channel_index = amp_data.index.get_level_values('channel')

    amp_data['y_locs'] = [plot_params['chan_locs'][ch][0] for ch in channel_index]
    amp_data['x_locs'] = [plot_params['chan_locs'][ch][1] for ch in channel_index]

    # http://stackoverflow.com/questions/15361143/how-to-fit-result-of-matplotlib-pyplot-contourf-into-circle

    N = 50    # size for interpolation, small number will make a dentate edge

    bottom,top = amp_data['y_locs'].min()-0.1, amp_data['y_locs'].max()+0.1 # set margin in the plot
    left,right = bottom,top
    corners = pd.DataFrame([[0,left,bottom], [0,right,bottom], [0,left,top], [0,right,top]],columns=['Amp','x_locs','y_locs'])
    amp_data = amp_data.append(corners)

    x, y, z = amp_data['y_locs'] - \
        left, amp_data['x_locs']-bottom, amp_data['Amp']

    xi = np.linspace(0,right-left, N)
    yi = np.linspace(0,top-bottom, N)
    zi = scipy.interpolate.griddata((x,y), z, (xi[None,:], yi[:,None]), method='cubic')

    # zi = get_topograph(data, plot_params['chan_locs'], channel_index, N)

    if 'zlim' in plot_params:
        zlim = plot_params['zlim']
        zi[zi<zlim[0]] = zlim[0]
        zi[zi>zlim[1]] = zlim[1]

    if 'mask' not in plot_params or plot_params['mask']:
        # make a circle mask
        # set points > radius to not-a-number. They will not be plotted.
        # the dr/2 makes the edges a bit smoother
        xy_center = [(right - left)/2,(top - bottom)/2]   # center of the plot
        radius = (amp_data['x_locs'].max() - amp_data['x_locs'].min())/2 + 0.05
        dr = xi[1] - xi[0]
        for i in range(N):
            for j in range(N):
                r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
                if (r - dr/2) > radius:
                    zi[j,i] = "nan"

        # draw a circle
        # change the linewidth to hide the 
        circle = matplotlib.patches.Circle(xy = xy_center, radius = radius-0.5, edgecolor = "k", facecolor = "none")
        ax.add_patch(circle)
        # draw a nose
        points = [(xy_center[0]-(radius-0.5)/5, xy_center[1]+radius-0.5), (xy_center[0], xy_center[1]+radius), (xy_center[0]+(radius-0.5)/5, xy_center[1]+radius-0.5)]
        line = plt.Polygon(points, closed=None, fill=None, edgecolor = "k", facecolor = "none")
        ax.add_patch(line)

    # # draw ears
    # points = [(xy_center[0]-(radius-0.5), xy_center[1]+(radius-0.5)/5), (xy_center[0]-radius, xy_center[1]), (xy_center[0]-(radius-0.5), xy_center[1]-(radius-0.5)/5)]
    # line = plt.Polygon(points, closed=None, fill=None, edgecolor = "k", facecolor = "none")
    # ax.add_patch(line)
    # points = [(xy_center[0]+(radius-0.5), xy_center[1]+(radius-0.5)/5), (xy_center[0]+radius, xy_center[1]), (xy_center[0]+(radius-0.5), xy_center[1]-(radius-0.5)/5)]
    # line = plt.Polygon(points, closed=None, fill=None, edgecolor = "k", facecolor = "none")
    # ax.add_patch(line)

    # use different number of levels for the fill and the lines
    if 'zlim' in plot_params:
        zlim = plot_params['zlim']
        CS = ax.contourf(xi, yi, zi, 60, cmap = plot_params['color'], zorder = 1,levels=np.linspace(zlim[0],zlim[1],40))
    else:
        CS = ax.contourf(xi, yi, zi, 60, cmap = plot_params['color'], zorder = 1)
    # contour line
    ax.contour(xi, yi, zi, colors = "grey", alpha = 0.5, zorder = 2, linestyles='solid')

    # add the data points, x[:-4] will drop the corner points we just added 
    ax.scatter(x[:-4], y[:-4], marker = 'o', c = 'b', alpha = 0.5, s = 2, zorder = 3)  

    # draw significant channel as white point
    if isinstance(pvalue_data, pd.Series):
        pvalue_data = pd.DataFrame(pvalue_data, columns=['pval'])
        pvalue_data['y_locs'] = amp_data['y_locs']
        pvalue_data['x_locs'] = amp_data['x_locs']

        for ind, i in pvalue_data.iterrows():
            if i.pval < plot_params['sig_limit']:
                ax.scatter(i.y_locs-bottom, i.x_locs-left,  c = 'w', s = 25, alpha=1, zorder = 3, linewidths=0.5, edgecolors='b')

    # remove the axis
    ax.axis('off')

def channel_locs(topo):
    # epochs.info['xy_locs']

    label = [i for i in topo.keys()]
    x = [i[0] for i in topo.values()]
    y = [i[1] for i in topo.values()]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(label):
        ax.annotate(txt, (x[i],y[i]))

    ax.axis('off')
    plt.show()
