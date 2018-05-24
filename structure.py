from .default import *
from . import io
from . import group

class Epochs():
    def __init__(self, epochs_data, montage_path='standard-10-5-cap385', info=dict()):
        if not isinstance(epochs_data, pd.DataFrame):
            raise Exception('Unsupported input!')

        if 'sample_rate' not in info:
            # inferred_freq = epochs_data.columns.inferred_freq
            # if len(inferred_freq)==1:
            #     inferred_freq = '1'+inferred_freq # sometimes inferred_freq is "L"
            # if inferred_freq[-1] == 'L':
            #     sr = int(1000/float(inferred_freq[:-1]))
            # elif inferred_freq[-1] == 'U':
            #     sr = int(1000000/float(inferred_freq[:-1]))
            info['sample_rate'] = 1000//(epochs_data.columns[1]-epochs_data.columns[0])

        info['subjects'] = {'all':list(epochs_data.index.get_level_values('subject').unique())}
        info['timepoints'] = {'all':list(epochs_data.columns)}

        info['conditions'] = dict()
        info['conditions']['all'] = list(epochs_data.index.get_level_values('condition').unique())

        info['channels'] = dict()
        info['channels']['all'] = list(epochs_data.index.get_level_values('channel').unique())
        
        info['trials'] = dict()
        info['trials']['all'] = list(epochs_data.index.get_level_values('trial').unique())

        for subj_id,subj_data in epochs_data.groupby(level=['subject']):
            info['conditions'][str(subj_id)] = sorted(list(subj_data.index.get_level_values('condition').unique()))
            info['channels'][str(subj_id)] = sorted(list(subj_data.index.get_level_values('channel').unique()))
            info['trials'][str(subj_id)] = sorted(list(subj_data.index.get_level_values('trial').unique()))

        if 'channel&id' not in info:
            info['channel&id'] = dict() # build up a dictionary contained id2channel and channel2id
            for ind,i in enumerate(info['channels']['all']):
                info['channel&id'][str(i)] = ind
                info['channel&id'][ind] = str(i)
           
        if 'xy_locs' not in info:
            info['xy_locs'] = io.load_topolocs(montage_path, info['channels']['all'])

        epochs_data_averaged = epochs_data.mean(level=list(np.setdiff1d(epochs_data.index.names, ['trial'])))
        epochs_data_averaged.index = pd.MultiIndex.from_tuples([(subject,condition,info['conditions']['all'].index(condition),channel) for (channel,condition,subject) in epochs_data_averaged.index],
                                 names=['subject','condition','trial','channel'])
        epochs_data_averaged.sort_index(inplace=True)

        self.all = epochs_data
        self.average = epochs_data_averaged
        self.info = info

    def extract(self, batch_script):
        batch_frame = group.parsing(batch_script, self)    
        return Extracted_epochs(self, batch_frame)

    def save(self, filepath, append=False):
        io.save_epochs(self, filepath, append=False, all_in_one=False)

class Extracted_epochs():
    def __init__(self, epochs, batch_frame):
        self.data = epochs
        self.frame = batch_frame
        self.info = epochs.info

    def iter_batchs(self):
        for batch_name, batch_frame in self.frame:
            yield batch_name, batch_frame

    def iter_cases(self, batch_frame):
        for case_frame in batch_frame:
            yield case_frame

    def get_batch_name(self):
        return self.frame.name

    def get_case_name(self, case_id=0):
        batch_name, batch_frame = self.frame
        case_frame = batch_frame[case_id]

        case_frame_dict = dict()
        for sub_case,sub_case_key in zip(case_frame,['subject','condition','trial','channel','timepoint']):
            case_frame_dict[sub_case_key+'_name'] = sub_case[0]

        return case_frame_dict

    def get_dataframe(self, case_id=0, average=True, to_print=False):
        batch_name, batch_frame = self.frame
        case_frame = batch_frame[case_id]

        if average:
            data_to_extracted = self.data.average
        else:
            data_to_extracted = self.data.all
        result = group.generate_case_data(case_frame, data_to_extracted)

        if to_print:
            print(f'batch_name: {batch_name}')
            print(f'case_name: {result.name}')
        return result

    def get_array(self, case_id=0, average=True, to_print=False):
        df = self.get_dataframe(case_id, average, to_print)
        return df.as_matrix()

    def get_index(self, case_id=0, average=True, to_print=False):
        df = self.get_dataframe(case_id, average, to_print)
        return df.index, df.columns

    def get_info(self, key):
        return self.info[key]

    def iter(self, mode='average'): # iterate the batch and the corresponding batchs, and apply the analyzing function
        def decorator(func):
            def wrapper(*args, **kw):
                data_to_extracted = getattr(self.data, mode) # e.g. self.data.average

                batch_name, batch_frame = self.frame
                analyzed_batch = []
                all_case_names = []

                for case_frame in batch_frame:
                    case_data = group.generate_case_data(case_frame, data_to_extracted)

                    all_case_names.append(case_data.name)
                    result = func(case_data, *args, **kw)

                    if isinstance(result, tuple): # if return value is not single
                        analyzed_batch.append(result)
                    else:
                        analyzed_batch.append(tuple([result]))

                all_case_names = np.array(all_case_names)
                name = ''
                for ind,i in enumerate(['subjects','conditions','trials','channels','timepoints']):
                    values = list(np.unique(all_case_names[:,ind]))
                    if values != ['All'] and set(values) != set(self.info[i]['all']):
                        name += ','.join(values) + ' '

                analyzed_batch_dfs = [] 
                # deal with the multiple return value(s) in the 'func', 
                # 'len(analyzed_batch[0])' refers to the number of return value(s)
                for i in range(len(analyzed_batch[0])):
                    if analyzed_batch[0][i] is None:
                        analyzed_batch_dfs.append(None)
                    else:
                        analyzed_batch_df = pd.concat([result[i] for result in analyzed_batch])
                        analyzed_batch_df.name = name
                        analyzed_batch_dfs.append(analyzed_batch_df)

                analyzed_batch = analyzed_batch_dfs

                if len(analyzed_batch) > 1:
                    return tuple(analyzed_batch[i] for i in range(len(analyzed_batch)))
                else:
                    return analyzed_batch[0]

            return wrapper
        return decorator

class Analyzed_data():
    def __init__(self, analysis_name, data, annotation=None, supplement=None, default_plot_params=dict()):
        if 'time_group' in data.columns.names and len(data.columns.get_level_values('time_group').unique()) == 1:
            data.columns = data.columns.get_level_values('time')

        self.analysis_name = analysis_name
        self.data = data
        self.annotation = annotation
        self.default_plot_params = default_plot_params
        self.supplement = supplement

    def __repr__(self):
        print('Name: ', self.analysis_name)
        print()
        print('**Samples in Data:')
        print(self.data.head())
        print()
        if isinstance(self.annotation, pd.DataFrame):
            print('**Samples in Annotation:')
            print(self.annotation.head())
        return ''

    def correct(self, on_annotation=False, method='fdr_bh'):
        from .statistics.stats_methods import multiple_comparison_correction
        if on_annotation:
            return Analyzed_data(self.analysis_name, self.data, multiple_comparison_correction(self.annotation, method=method), self.supplement, self.default_plot_params)
        else:
            return Analyzed_data(self.analysis_name, multiple_comparison_correction(self.data, method=method), self.annotation, self.supplement, self.default_plot_params)

    def save(self, filepath):
        io.save_result(self, filepath)


