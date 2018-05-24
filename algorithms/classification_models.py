from ..default import *
from .. import structure
from .basic import * 
from ..statistics import stats_methods

import types

import sklearn
from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

def reshape_X(X, extra_params):
    X = np.array([np.array(x) for x in X])
    return X

def reshape_Y(Y, extra_params):
    return sklearn.preprocessing.LabelBinarizer().fit_transform(Y)

def sklearn_model(X, Y, train_index, test_index, model):
    model.fit(X[train_index], Y[train_index])

    if getattr(model,'predict_proba', None):
        prob_train = model.predict_proba(X[train_index])[:,1]
        prob_test = model.predict_proba(X[test_index])[:,1]
    else:
        prob_train = model.decision_function(X[train_index])
        prob_test = model.decision_function(X[test_index])

    score_train = roc_auc_score(Y[train_index],prob_train)
    score_test = roc_auc_score(Y[test_index],prob_test)

    if hasattr(model, 'coef_'): # weights
        weights = model.coef_[0] # n*1 matrix, so [0] makes it n vector
    else:
        weights = None
    return score_train, score_test, weights

def group_stats(df, reshape_X_method, reshape_Y_method, run_model, fold, test_size, extra_params=dict()):
    scores_to_test = dict()
    weights = dict()
    has_weights = True
    baselines = []
    for subject_group_id,subject_group_data in df.groupby(level='subject'):  # df: channel,condition_group,trial
        X, Y = zip(*[(data.T, name[0]) for name,data in subject_group_data.groupby(level=['condition_group','trial'])])
        channels = X[0].index.get_level_values('channel')
        
        X, Y = reshape_X_method(X, extra_params), reshape_Y_method(Y, extra_params)

        score_train_folds = []
        score_test_folds = []
        weight_folds = []
        for train_index, test_index in StratifiedShuffleSplit(fold, test_size=test_size).split(X, Y):
            if isinstance(run_model, types.FunctionType):
                score_train, score_test, weight_trained = run_model(X, Y, train_index, test_index, extra_params)
            else:
                score_train, score_test, weight_trained = sklearn_model(X, Y, train_index, test_index, run_model)

            score_train_folds.append(score_train)
            score_test_folds.append(score_test)
            if weight_trained is None:
                has_weights = False
            else:
                weight_folds.append(weight_trained)

        scores_to_test[subject_group_id] = np.mean(score_test_folds)
        if has_weights:
            weights[subject_group_id] = {k:v for k,v in zip(channels, np.abs(np.mean(weight_folds, axis=1)))}
        
        baselines.append(1/len(np.unique(Y)))

    # t,pvalue = scipy.stats.ttest_rel(scores_to_test, baselines)
    pvalue, t = two_sample(list(scores_to_test.values()), baselines, reps=1000, stat=lambda u, v: np.mean(u-v), alternative='greater')

    scores_to_test = pd.Series(scores_to_test)
    scores_to_test.index.name = 'subject'

    if has_weights:
        weights = pd.DataFrame(weights)
        weights.index.name = 'channel'
        weights.columns.name = 'subject'
    else:
        weights = None

    return pvalue, scores_to_test, weights

def classification(self, step_size='1ms', win_size='1ms', sample='mean', run_model=linear_model.LogisticRegression(class_weight="balanced"), reshape_X_method=reshape_X, reshape_Y_method=reshape_Y, fold=30, test_size=0.3, extra_params=dict(), parallel=False):
    # with the decorator, we can just focuse on case data instead of batch/collection data
    @self.iter('all')
    def learn(case_raw_data):
        case_raw_data = sampling(case_raw_data, step_size, win_size, sample)

        case_raw_data.columns = case_raw_data.columns.get_level_values(1)  # remvoe "time_group"
        check_availability(case_raw_data, 'condition_group', '==2') 

        return roll_on_levels(case_raw_data, group_stats, arguments_dict=dict(run_model=run_model, reshape_X_method=reshape_X_method, reshape_Y_method=reshape_Y_method, fold=fold, test_size=test_size, extra_params=extra_params), levels='time', prograssbar=True, parallel=parallel)

    pvalue_batch, score_batch, weight_batch = learn()

    default_plot_params = dict(title='Pattern classification',plot_type=['direct','heatmap'], x_len=12, re_assign=[(0,0.01,0.05,1),(3,2,1)], color=sns.cubehelix_palette(light=1, as_cmap=True), grid=True,  x_title='time', y_title='condition_group',cbar_title='pvalue',cbar_values=['>=0.05','<0.05','<0.01'])

    return structure.Analyzed_data('Pattern classification', pvalue_batch, score_batch, [weight_batch], default_plot_params=default_plot_params)
