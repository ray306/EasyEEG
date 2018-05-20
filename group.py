from .default import *

Case = namedtuple('Case',['name','val'])
Batch = namedtuple('Batch',['name','val'])

def parsing(batch_script, epochs_data):
    def conditions_filter_by_factor(query_str):
        cond_format = epochs_data.info['conditions']['format']
        # if there is anything between brackets
        if len(re.sub(r'\(.*?\)','',cond_format))>0: 
            cond_format = cond_format.replace('(','(?P<c_').replace(')','>.*)')
        else:
            cond_format = cond_format.replace('(','(?P<c_').replace(')','>.)')
        # generate the dataframe of factors    
        condition_df = []
        for condition in epochs_data.info['conditions']['all']:
            pattern = re.search(cond_format, condition)
            if pattern:
                indicators = pattern.groupdict()
                indicators['name'] = condition
                condition_df.append(indicators)
        if condition_df!=[]:
            condition_df = pd.DataFrame(condition_df)
        else:
            raise Exception(f'Cannot find any thing by the format "{epochs_data.info["conditions"]["format"]}"')
        query_str = query_str.replace('|','||').replace('&','&&')+'$'
        query_str = re.sub(r'(^|\W)(\w*?)!=(.*?)([\W])',r'\1c_\2!="\3"\4',query_str)
        query_str = re.sub(r'(^|\W)([ a-zA-Z0-9]*?)=(.*?)([\W])',r'\1c_\2=="\3"\4',query_str)
        query_str = re.sub(r'!c_==""',r'!=',query_str)
        query_str = query_str[:-1].replace('||','|').replace('&&','&')
        return condition_df.query(query_str).name.tolist()

    def group_by_list(li,levels):
        # li = ['aB1','aC1','bB1','bC1','aB2','aC2','bB2','bC2']
        # group_by_list(li,[1,2])
        
        def groupBy(li,level):
            li.sort(key=lambda x:x[level])
            return [list(sub_li) for i, sub_li in itertools.groupby(li,lambda x:x[level])]
        
        if len(levels)>0:
            return [group_by_list(sub_li,levels[1:]) for sub_li in groupBy(li,levels[0])]
        else:
            return li

    def case_parser(key, case_str):
        def ambigous_symbol_converter(old_str):
            while 1:
                new_str = re.sub(r'{([^}]*?)\&(.*?)}', '{\g<1><and>\g<2>}',old_str)
                new_str = re.sub(r'{([^}]*?)\|(.*?)}', '{\g<1><or>\g<2>}',new_str)
                if old_str==new_str:
                    return new_str
                else:
                    old_str = new_str

        def plus_minus(text): # make the PLUS mix and MINUS mix
            text = '+' + text.replace(' ','')
            mix = re.findall(r'[\+\-]{,2}[^\+\-]+',text)
            plus,minus=[],[]
            for c in mix:   
                if c[0]=='+':
                    plus.append(c[1:])
                elif c[0]=='-':
                    minus.append(c[1:])
            if plus==[]: plus = slice(None)
            return {'+':plus,'-':minus} 

        def timepoint_parser(timepoint_str_list):
            interval = 1000//epochs_data.info['sample_rate']
            timepoint_list = []
            for timepoint_str in timepoint_str_list:
                if '~' in timepoint_str:
                    start,end = timepoint_str.split('~')
                    timepoint_list += list(np.arange(int(start),int(end)+interval,interval).astype(int))
                else:
                    timepoint_list.append(int(timepoint_str))

            return timepoint_list

        def subject_parser(subject_str_list):
            subject_list = [int(i) for i in subject_str_list]
            return subject_list

        def trial_parser(trial_str_list):
            trial_list = []
            for trial_str in trial_str_list:
                if '~' in trial_str:
                    start,end = trial_str.split('~')
                    trial_list += list(range(int(start),int(end)+1))
                else:
                    trial_list.append(int(trial_str))
            return trial_list

        def condition_parser(condition_str_list):
            # epochs.info['conditions']['format'] = '(global)_(local)_(target)'
            # 'a:{global=T&target=G}&{global=F&target=G}'

            condition_list = []
            for condition_str in condition_str_list:
                if condition_str[0] is '{' and condition_str[-1] is '}' :
                    query_str = condition_str[1:-1].replace('<and>','&').replace('<or>','|')
                    condition_str = conditions_filter_by_factor(query_str)
                else:
                    condition_str = [condition_str]
                condition_list += condition_str
            return condition_list

        if ':' in case_str:
            name,val = case_str.split(':')
        else:
            name,val = case_str,case_str

        val = ambigous_symbol_converter(val)
        val = [plus_minus(i) for i in val.split('&')]
        
        if key=='timepoints':
            val = [{'+':timepoint_parser(part['+']),'-':timepoint_parser(part['-'])}
                    for part in val]
        elif key=='trials':
            val = [{'+':trial_parser(part['+']),'-':trial_parser(part['-'])}
                        for part in val]
        elif key=='conditions':
            val = [{'+':condition_parser(part['+']),'-':condition_parser(part['-'])}
                        for part in val]
        elif key=='subjects':
            val = [{'+':subject_parser(part['+']),'-':subject_parser(part['-'])}
                        for part in val]
        return Case(name,val)

    def batch_str_parser(batch_str):
        batch = dict()
        batch_str_list = batch_str.split('@')

        batch['conditions'] = batch_str_list[0].strip()
        if len(batch_str_list)>0:
            chs = batch_str_list[1].strip()
            if chs!='':
                batch['channels'] = chs
        if len(batch_str_list)>1:
            tps = batch_str_list[2].strip()
            if tps!='':
               batch['timepoints'] = tps
        return batch

    def combine(key,sep):
        li = epochs_data.info[key]['all']
        if sep=='+':
            return ['All:' + sep.join([str(i) for i in li])]
        elif sep==',':
            return [str(i) for i in li]

    # convert the description into dict if the description is a string
    if isinstance(batch_script, str):
        batch_script = batch_str_parser(batch_script)
    elif not isinstance(batch_script, dict):
        raise Exception(f'The description "{batch_script}" should be a string or dict!')
    
    # convert into full-description dict
    batch_script_full = dict()
    for key in ['subjects', 'conditions', 'trials', 'channels', 'timepoints']:
        if key in batch_script:
            if batch_script[key] == 'each':
                batch_script_full[key] = combine(key,',')
            else:
                if isinstance(batch_script[key], str):
                    batch_script_full[key] = batch_script[key].split(',')
                else:
                    batch_script_full[key] = batch_script[key]
        else:
            batch_script_full[key] = combine(key,'+')

        # parse the string of case
        batch_script_full[key] = [case_parser(key,i) for i in batch_script_full[key]]
    
    batch_name = []
    for batch_k,batch_v in batch_script_full.items():
        batch_desp = ','.join([case_v.name for case_v in batch_v])
        if batch_desp!='All':
            batch_name.append(batch_desp)

    # Get the cartesian product of the dict
    batch_frame = []
    for case_frame in list(itertools.product(*batch_script_full.values())):
        batch_frame.append(case_frame)

    batch_frame = Batch(batch_name, batch_frame)

    return batch_frame

def filter(epochs_data, subject=slice(None), condition=slice(None), trial=slice(None), channel=slice(None), timepoint=slice(None), filter_dict=dict()):
    if 'subject' in filter_dict:
        subject = filter_dict['subject']
    if 'condition' in filter_dict:
        condition = filter_dict['condition']
    if 'trial' in filter_dict:
        trial = filter_dict['trial']
    if 'channel' in filter_dict:
        channel = filter_dict['channel']
    if 'timepoint' in filter_dict:
        timepoint = filter_dict['timepoint']

    return epochs_data.loc[ids[subject,condition,trial,channel],ids[timepoint]]

def generate_case_data(case_frame, epochs_data_todo):

    case_frame_dict = dict()
    for sub_case,sub_case_key in zip(case_frame,['subject','condition','trial','channel','timepoint']):
        case_frame_dict[sub_case_key+'_name'] = sub_case[0]
        case_frame_dict[sub_case_key] = sub_case[1]

    get_all = lambda l: sum([i['+']+i['-'] for i in l], [])
    extracted_data = epochs_data_todo.loc[
            ids[get_all(case_frame_dict['subject']),
                get_all(case_frame_dict['condition']),
                get_all(case_frame_dict['trial']),
                get_all(case_frame_dict['channel'])
               ],ids[get_all(case_frame_dict['timepoint'])]]

    # subtract baselines
    def subtract(data,baseline,level):
        if target_level == 'timepoint':
            baseline = baseline.mean(axis=1)
            return data.subtract(baseline,axis=0)
        else:
            name_d = {'subject':0,'condition':0,'trial':0,'channel':0}

            levels_keeped = ['subject','condition','channel']
            if 'trial' != level:
                levels_keeped.remove(level)

            baseline = baseline.mean(level=levels_keeped)
            baseline_dict = dict((ind,list(i)) for ind,i in baseline.iterrows())
            
            def sub(x):
                name_d['subject'],name_d['condition'],name_d['trial'],name_d['channel'] = x.name
                ind = tuple(name_d[i] for i in levels_keeped)
                return x-baseline_dict[ind]

            return data.apply(sub,axis=1)

    levels = ['subject','condition','trial','channel','timepoint']
    for target_level in levels:
        data_t = []
        for group_idx,group in enumerate(case_frame_dict[target_level]):
            group_data = filter(extracted_data, filter_dict={target_level:group['+']})

            if len(group['-'])>0:
                baseline = filter(extracted_data, filter_dict={target_level:group['-']})
                group_data = subtract(group_data, baseline, target_level)

            if target_level == 'timepoint':
                group_data.columns = pd.MultiIndex.from_tuples([(group_idx,i) for i in group_data.columns])
            else:
                group_data.index.set_levels(
                                [f'{group_idx} {i}' for i in group_data.index.levels[levels.index(target_level)]],
                                        target_level,inplace=True)
                                    
            data_t.append(group_data)

        extracted_data = pd.concat(data_t)
        extracted_data.sort_index(inplace=True)
        
    for target_level in levels[:-1]:
        # extracted_data[target_level+'_group'] = [i.split(' ')[0] for i in extracted_data.index.get_level_values(target_level)]
        extracted_data[target_level+'_group'] = [i.split(' ')[0]+' '+case_frame_dict[target_level+'_name'] for i in extracted_data.index.get_level_values(target_level)]
        extracted_data.set_index(target_level+'_group', append=True, inplace=True)
        
        extracted_data.columns.set_levels(list(extracted_data.columns.levels[1][:-1]), level=1, inplace=True)
        
        extracted_data.index.set_levels(
                    [' '.join(i.split(' ')[1:]) for i in extracted_data.index.levels[levels.index(target_level)]], 
                    target_level,inplace=True)

    extracted_data.columns.names = ['time_group','time']
    extracted_data.name = [case_frame_dict[target_level+'_name'] for target_level in levels]
    return extracted_data
