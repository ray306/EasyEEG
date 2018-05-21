from ..default import *

def multiple_comparison_correction(pvs, method='fdr_bh'):
    if method == 'cluster':
        def correct(v):
            level = 0.05
            v = np.concatenate(([1],v,[1]))
            for idx in range(1,len(v)-1):
                if v[idx] < level and v[idx-1] >= level and v[idx+1] >= level:
                    v[idx] = level
            return v[1:-1]
    else:
        correct = lambda v: statsmodels.sandbox.stats.multicomp.multipletests(v, 0.05, method)[1]
    
    pvs_new = pd.DataFrame([correct(i) for i in pvs.values], index=pvs.index, columns=pvs.columns)
    pvs_new.name = pvs.name
        
    return pvs_new

def t_test(values):
    a,b = values
    t, pv = scipy.stats.ttest_rel(a,b)
    return {'pvalue':pv, 'effect':t}

def permutation_test(values, reps=1000, alternative='two-sided'):
    values = np.array(values)
    if len(values)==2:
        pv,t = two_sample(values[0],values[1], reps=reps,stat=lambda u,v: mean(u-v),alternative='two-sided')
        return {'pvalue':pv, 'effect':t}
    else:
        return None

def get_pvalue_from_distribution(result_real, dist_baseline):
    dist_baseline.append(result_real)
    dist_baseline.sort()
    
    pvalue = 1-dist_baseline.index(result_real)/len(dist_baseline)
    if pvalue == 0:
        pvalue += 1/len(dist_baseline)

    return pvalue

def permutation_on_condition(data,method,shuffle_count=1000):
    def condition_shuffled(data):
        group_labels = list(data.index.get_level_values(level='cond_group'))
        random.shuffle(group_labels)
        data.index = data.index.set_labels(group_labels,level='cond_group')
        return data
    # keep
    val_raw = method(data)

    # shuffle
    baseline =[]
    for i in range(shuffle_count):
        val_shuffled = method(condition_shuffled(data))
        baseline.append(val_shuffled)

    baseline.append(2) # correction
    baseline.append(val_raw)
    baseline.sort()

    pv = 1-baseline.index(val_raw)/shuffle_count

    return pv,None

def fdr(pvs):
    re_calc = lambda v: statsmodels.sandbox.stats.multicomp.multipletests(v, 0.05, 'fdr_bh')[1]
    if isinstance(pvs, pd.DataFrame):
        fdr = [re_calc(i) for i in pvs.values]
        pvs_new = pd.DataFrame(fdr, index=pvs.index, columns=pvs.columns)
    elif isinstance(pvs, pd.Series):
        fdr = re_calc(pvs.values)
        pvs_new = pd.Series(fdr, index=pvs.index, name=pvs.name)
    elif isinstance(pvs, dict):
        fdr = re_calc(pvs.values())
        pvs_new = dict(zip(pvs.keys(),fdr))
    elif isinstance(pvs, list):
        pvs_new = re_calc(pvs)
    else:
        raise ValueError('Only support pd.DataFrame, pd.Series,dict, and list')
    return pvs_new

def anova():
    pass

def similarity():
    pass

def classifier():
    pass            
