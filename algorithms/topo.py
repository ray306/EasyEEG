from ..default import *
from .. import structure
from .basic import * 
from ..statistics import stats_methods

def topography(self, step_size='1ms', win_size='1ms', sample='mean', sig_limit=0.05):
    @self.iter('average')
    def to_topo(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)
            
        data_with_subject = case_raw_data.mean(level=['subject','condition_group','channel'])
        if len(data_with_subject.index.get_level_values('condition_group').unique()) == 2:  
            condition_groups_data = [data for idx,data in data_with_subject[0].groupby(level='condition_group')]
            
            topo_result = condition_groups_data[0].mean(level='channel') - condition_groups_data[1].mean(level='channel')
            recover_index(topo_result, data_with_subject,'condition_group') # re-add level 'condition_group' in index
            stats_result = roll_on_levels_and_compare(data_with_subject, stats_methods.t_test, levels=['time','channel'], 
                              between='condition_group', in_group='subject',prograssbar=False)
            # [0] is ugly
            result = pd.concat([topo_result, stats_result[0]],
                               keys=['Amp', 'p_val'], axis=1)

        else:
            result = data_with_subject.mean(level=['condition_group','channel'])
            result.columns = pd.MultiIndex.from_product([['Amp'], result[0].columns],names=['','time'])

        return result

    topo_collection = to_topo()

    minmax = [(t.min().min(),t.max().max()) for t in topo_collection]
    minmax = (np.array(minmax).min(),np.array(minmax).max())

    default_plot_params = dict(title='Topography',plot_type=['matrix','topograph'], zlim=minmax, color=plt.cm.jet, cbar_title='uV',
        chan_locs=self.info['xy_locs'], sig_limit=sig_limit, x_title='time', y_title='condition_group')
    return structure.Analyzed_data('Topography', topo_collection, default_plot_params=default_plot_params)
    
def significant_channels_count(self, step_size='1ms', win_size='1ms', sample='mean', sig_limit=0.05):
    @self.iter('average')
    def to_signif(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)
            
        data_with_subject = case_raw_data.mean(level=['subject','condition_group','channel'])
        
        check_availability(data_with_subject, 'condition_group', 2)

        stats_result = roll_on_levels_and_compare(data_with_subject, stats_methods.t_test, levels=['time','channel'], 
                          between='condition_group', in_group='subject',prograssbar=True)
  
        stats_result = stats_result.stack('time').unstack('channel').apply(lambda x:sum(x<sig_limit),axis=1).unstack('time')

        return stats_result

    signif_collection = to_signif()

    default_plot_params = dict(title='significant_channels_count',plot_type=['direct','heatmap'], x_len=12,
                                color=sns.cubehelix_palette(light=1, as_cmap=True), x_title='time', y_title='condition_group',cbar_title='Count')
    return structure.Analyzed_data('significant_channels_count', signif_collection, default_plot_params=default_plot_params)
    
