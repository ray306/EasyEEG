from ..default import *
from .. import structure
from .basic import * 
from ..statistics import stats_methods
from scipy import signal

def topography(self, step_size='1ms', win_size='1ms', sample='mean', sig_limit=0.05):
    @self.iter('average')
    def to_topo(case_raw_data):
        check_availability(case_raw_data, 'channel', '>1')

        data_with_subject = case_raw_data.mean(level=['subject', 'condition_group', 'channel'])
        data_with_subject = sampling(data_with_subject, step_size, win_size, sample)

        data_with_subject.columns = data_with_subject.columns.get_level_values(1)  # remvoe "time_group"
        
        if len(data_with_subject.index.get_level_values('condition_group').unique()) == 2:
            topo_result = subtract(data_with_subject, between='condition_group', align='channel')

            stats_result = roll_on_levels_and_compare(data_with_subject, stats_methods.t_test, levels=['time', 'channel'], between='condition_group', in_group='subject', prograssbar=False)
            return topo_result, stats_result

        else:
            topo_result = data_with_subject.mean(level=['condition_group', 'channel'])
            return topo_result, None
    
    topo_batch, stats_data = to_topo()

    minmax = (topo_batch.min().min(), topo_batch.max().max())

    default_plot_params = dict(title='Topography', plot_type=['matrix', 'topograph'], zlim=minmax, color=plt.cm.jet, cbar_title='uV', chan_locs=self.info['xy_locs'], sig_limit=sig_limit, x_title='time', y_title='condition_group')

    return structure.Analyzed_data('Topography', topo_batch, stats_data, default_plot_params=default_plot_params)

def frequency_topography(self, step_size='1ms', win_size='1ms', sample='mean', sig_limit=0.05, target=10):
    if isinstance(target, (int, float)):
        freqs = [target]
    elif isinstance(target, list) and len(target) == 2 \
            and isinstance(target[0], (int, float)) and isinstance(target[1], (int, float)):
        freqs = np.arange(target[0], target[1])
    else:
        raise Exception(
            'Unsupported value for "target". The value should be a number, or a list of two numbers')
    
    if freqs[0] == 0: freqs += 0.001
    
    def cwt(name, data):
        cwt_result = signal.cwt(
            np.array(data)[0], signal.ricker, widths=freqs).mean(axis=0)
        cwt_result = pd.DataFrame(
            [cwt_result], index=data.index, columns=data.columns)
        return cwt_result

    @self.iter('average')
    def to_topo(case_raw_data):
        data_without_subject = convert(case_raw_data, ['condition_group', 'channel'], cwt)
        data_without_subject = sampling(data_without_subject, step_size, win_size, sample)

        data_without_subject.columns = data_without_subject.columns.get_level_values(1)  # remvoe "time_group"

        if len(data_without_subject.index.get_level_values('condition_group').unique()) == 2:
            data_with_subject = convert(case_raw_data, ['subject', 'condition_group', 'channel'], cwt)
            data_with_subject = sampling(data_with_subject, step_size, win_size, sample)

            data_with_subject.columns = data_with_subject.columns.get_level_values(1)  # remvoe "time_group"  

            topo_result = subtract(
                data_with_subject, between='condition_group', align='channel')

            stats_result = roll_on_levels_and_compare(data_with_subject, stats_methods.t_test, levels=['time', 'channel'], between='condition_group', in_group='subject', prograssbar=False)

            return topo_result, stats_result
        else:
            topo_result = data_without_subject
            return topo_result, None

    topo_batch, stats_data = to_topo()
    minmax = (topo_batch.min().min(), topo_batch.max().max())

    default_plot_params = dict(title='Topography', plot_type=['matrix', 'topograph'], zlim=minmax, color=plt.cm.jet, cbar_title='Power', chan_locs=self.info['xy_locs'], sig_limit=sig_limit, x_title='time', y_title='condition_group')

    return structure.Analyzed_data('Topography', topo_batch, stats_data, default_plot_params=default_plot_params)

def significant_channels_count(self, step_size='1ms', win_size='1ms', sample='mean', sig_limit=0.05):
    @self.iter('average')
    def to_signif(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)
        data_with_subject = case_raw_data.mean(level=['subject','condition_group','channel'])
        check_availability(data_with_subject, 'condition_group','==2')

        stats_result = roll_on_levels_and_compare(data_with_subject, stats_methods.t_test, levels=['time','channel'], between='condition_group', in_group='subject',prograssbar=True)
  
        stats_result = stats_result.stack('time').unstack('channel').apply(lambda x:sum(x<sig_limit),axis=1).unstack('time')

        return stats_result

    signif_batch = to_signif()

    default_plot_params = dict(title='significant_channels_count',plot_type=['direct','heatmap'], x_len=12, color=sns.cubehelix_palette(light=1, as_cmap=True), x_title='time', y_title='condition_group',cbar_title='Count')

    return structure.Analyzed_data('significant_channels_count', signif_batch, default_plot_params=default_plot_params)
    
