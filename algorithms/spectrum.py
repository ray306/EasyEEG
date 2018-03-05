from ..default import *
from .. import structure
from .basic import * 
from ..statistics import stats_methods

import scipy.fftpack
from scipy import signal

comparison_params = dict(test=stats_methods.t_test, win='1ms', method='mean', sig_limit=0.05, need_fdr=False)

def Spectrum(self, compare=False, freq_span=(0,30), target='power', comparison_params=comparison_params):
    # with the decorator, we can just focuse on case data instead of batch/collection data
    @self.iter('average')
    def to_spetrum(case_raw_data):
        erp = case_raw_data.mean(level=['subject','condition_group','channel_group'])

        Spectrum_df = []
        for name,data in erp.groupby(level=['subject','condition_group','channel_group']):
            N = data.shape[1]

            fft_result = scipy.fftpack.fft(data,axis=1)
            if target=='power':
                fft_result = 2.0/N * np.abs(fft_result)
            elif target=='phase':
                fft_result = np.angle(fft_result)
            else:
                raise Exception('Please set the parameter "target" as "power", or "phase".')
            fft_result = fft_result[:,:N//2]
            if freq_span[0]==0:
                fft_result[:,0] = 0

            index = pd.MultiIndex.from_tuples([name], 
                names=['subject','condition_group','channel_group'])

            fft_result = pd.DataFrame(fft_result,
                index=index,
                columns=np.linspace(0, self.info['sample_rate'], N//2)) # resolution: sr/N
            fft_result = fft_result.loc[:, freq_span[0]:freq_span[1]] 
            
            fft_result.columns = pd.MultiIndex.from_tuples([(0, freq) for freq in fft_result.columns], names=['','frequency'])
            Spectrum_df.append(fft_result)
        
        Spectrum_df = pd.concat(Spectrum_df)
        return Spectrum_df
    
    spetrum_collection = to_spetrum()
    if compare:
        stats_data = [stats_compare(spetrum_batch, comparison_params, levels='frequency', between='condition_group', in_group='subject') 
                        for spetrum_batch in spetrum_collection]
    else:
        stats_data = None

    default_plot_params = dict(plot_type=['direct','spectrum'], y_title='Spectrum', err_style='ci_band', color="Set1", style='darkgrid',win=comparison_params['win'],sig_limit=0.05)
    return structure.Analyzed_data('Spectrum', spetrum_collection, stats_data, default_plot_params)

# grand average
def Time_frequency(self, freq_span=(0,30)):
    # with the decorator, we can just focuse on case data instead of batch/collection data
    @self.iter('average')
    def to_tf(case_raw_data):
        erp = case_raw_data.mean(level=['condition_group','channel_group'])

        tf_df = []
        for name,data in erp.groupby(level=['condition_group','channel_group']):
            freqs = np.arange(freq_span[0], freq_span[1])
            if freq_span[0]==0:
                widths = freqs+0.001
            else:
                widths = freqs
            # index = pd.MultiIndex.from_tuples([(*name,freq) for freq in freqs[::-1]], 
            #     names=['condition_group','channel_group','frequency'])
    
            cwt_result = signal.cwt(np.array(data)[0], signal.ricker, widths=widths)
            cwt_result = pd.DataFrame(cwt_result,index=freqs[::-1],columns=data.columns)

            tf_df.append(cwt_result)
        
        tf_df = pd.concat(tf_df)
        return tf_df
    
    tf_collection = to_tf()

    default_plot_params = dict(plot_type=['direct','heatmap'], x_title='time', y_title='frequency', 
        color="RdBu_r", style='white', grid=False, cbar_title='Power')
    return structure.Analyzed_data('Time Frequency', tf_collection, None, default_plot_params)
