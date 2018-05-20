from ..default import *
from .. import structure
from .basic import * 
from ..statistics import stats_methods

from scipy.spatial.distance import cosine # 1-cosD

def calc_cosD(df): # df: channel,subject,condition_group
    cond_A, cond_B = [average(conditon_group_data, keep='channel')
        for conditon_group_id,conditon_group_data in df.groupby(level='condition_group')]
    return cosine(cond_A, cond_B)

def sub_func(group_data, shuffle=500, within_subject=True):
    result_real = calc_cosD(group_data)
    dist_baseline = []
    for _ in range(shuffle):
        shuffle_on_level(group_data, 'condition_group', within_subject=within_subject)
        dist_baseline.append(calc_cosD(group_data))

    pvalue = stats_methods.get_pvalue_from_distribution(result_real, dist_baseline)
    return pvalue, result_real

def tanova(self,step_size='1ms',win_size='1ms',sample='mean',shuffle=500,strategy=1,parallel=False):
    # with the decorator, we can just focuse on case data instead of batch/collection data
    @self.iter('all')
    def to_tanova1(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)
        case_raw_data.columns = case_raw_data.columns.get_level_values(1)  # remvoe "time_group"
        check_availability(case_raw_data, 'condition_group', '==2')
        return roll_on_levels(case_raw_data, sub_func, arguments_dict=dict(shuffle=shuffle, within_subject=False), levels='time', prograssbar=True, parallel=parallel)

    @self.iter('all')
    def to_tanova2(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)
        case_raw_data.columns = case_raw_data.columns.get_level_values(1)  # remvoe "time_group"
        check_availability(case_raw_data, 'condition_group', '==2')
        return roll_on_levels(case_raw_data, sub_func, arguments_dict=dict(shuffle=shuffle, within_subject=True), levels='time', prograssbar=True, parallel=parallel)

    @self.iter('average')
    def to_tanova3(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)
        case_raw_data.columns = case_raw_data.columns.get_level_values(1)  # remvoe "time_group"
        check_availability(case_raw_data, 'condition_group', '==2')
        return roll_on_levels(case_raw_data, sub_func, arguments_dict=dict(shuffle=shuffle, within_subject=True), levels='time', prograssbar=True, parallel=parallel)

    if strategy==1:
        tanova_collection, annotation_collection = to_tanova1()
    elif strategy==2:
        tanova_collection, annotation_collection = to_tanova2()
    elif strategy==3:
        tanova_collection, annotation_collection = to_tanova3()

    default_plot_params = dict(title='TANOVA',plot_type=['direct','heatmap'], x_len=12, re_assign=[(0,0.01,0.05,0.1,1),(4,3,2,1)], color=sns.cubehelix_palette(light=1, as_cmap=True), grid=True, x_title='time', y_title='condition_group',cbar_title='pvalue',cbar_values=['>0.1','<0.1','<0.05','<0.01'])

    return structure.Analyzed_data('TANOVA', tanova_collection, annotation_collection, default_plot_params=default_plot_params)

def cosine_distance_dynamics(self):
    # with the decorator, we can just focuse on case data instead of batch/collection data
    @self.iter('average')
    def calc(case_raw_data):
        check_availability(case_raw_data, 'condition_group', '==2')
        return roll_on_levels(case_raw_data, calc_cosD, levels='time')

    cosine_distance_collection = calc()

    default_plot_params = dict(title='cosine_distance_dynamics', plot_type=['direct','waveform'], x_title='time', y_title='distance', color="Set1", style='darkgrid')
    return structure.Analyzed_data('cosine distance dynamics', cosine_distance_collection, default_plot_params=default_plot_params)
