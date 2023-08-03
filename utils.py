import scipy
import json as js
import numpy as np
import xarray as xr
import pickle as pkl
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from numpy.lib.recfunctions import append_fields
from pybeh.temp_fact import temp_fact
from pybeh.dist_fact import dist_fact

from constants import subjects_powerfile_elsewhere, ROIs, non_peripheral_egi, non_peripheral_biosemi

def get_eeg_sys(subject):
    sys = 'biosemi' if int(subject[-3:])>330 else 'egi'
    return sys

def load_feature(subject, 
                 task_phase,
                 ret_type='ci',
                 exclude_sessions=None, 
                 exclude_freq=None, 
                 exclude_peripheral_channels=True, 
                 stack_feature=True):
    
    '''
    loads the z-scored power features for encoding events
    
    :param task_phase: 'enc' or 'ret', encoding or retrieval
    :param ret_type: ci, cp, ce, or pe, determines the ret_label column being added
    :param exclude_sessions: list of session numbers to exclude
    :param exclude_freq: list of frequencies to exclude
    :param exclude_peripheral_channels: exclude channels close to face/neck
    :param feature_stack: stack frequency and channel dimensions and returns shape [events, feature]
    
    '''
    
    def prepare_ret_labels(ret_power, ret_type):

        new_labels = None

        if ret_type == 'ci': # correct vs. intrusion, keep all events
            new_labels = np.zeros(len(ret_power.events.values))
            new_labels[ret_power.events.values['intrusion']==0] = 1
        elif ret_type == 'cp': # correct vs. pli
            ret_power = ret_power.sel(events=ret_power.events.values['intrusion']>=0)
            new_labels = np.zeros(len(ret_power.events.values))
            new_labels[ret_power.events.values['intrusion']==0] = 1
        elif ret_type == 'ce': # correct vs. eli
            ret_power = ret_power.sel(events=ret_power.events.values['intrusion']<=0)
            new_labels = np.zeros(len(ret_power.events.values))
            new_labels[ret_power.events.values['intrusion']==0] = 1
        elif ret_type == 'pe': # pli vs. eli
            ret_power = ret_power.sel(events=ret_power.events.values['intrusion']!=0)
            new_labels = np.zeros(len(ret_power.events.values))
            new_labels[ret_power.events.values['intrusion']>0] = 1

        ret_power.coords['events'] = append_fields(ret_power.events.values, 
                                                   'rec_label', new_labels, usemask=False)

        return ret_power
    
    power_file_root = 'scratch/mtpower/' if subject in subjects_powerfile_elsewhere else '/scratch/liyuxuan/ltpFR2/mtpower/'

    f = power_file_root + '%s/%s_%s_zpower.pkl' % (subject, subject, task_phase)
    x = pkl.load(open(f, 'rb'))
    
    if exclude_sessions is not None:
        x = x.sel(events = ~np.in1d(x.events.values['session'], exclude_sessions) )
    if exclude_freq is not None:
        x = x.sel(frequency = ~np.in1d(x.frequency.values, exclude_freq) )
    if exclude_peripheral_channels:
        non_peripheral = non_peripheral_biosemi if int(subject[-3:]) > 330 else non_peripheral_egi
        chanlist = [ch for ch in x.channels.values if ch in non_peripheral]
        x = x.sel(channels=chanlist)
    if stack_feature:
        x = x.stack(features=('frequency', 'channels'))
    if task_phase=='ret':
        x = prepare_ret_labels(x, ret_type) # adds a binary rec_label column
    
    return x

def resample_enc_events(feature):
    events = feature.events.values.copy()
    resampled_events_mstime = []
    for sess in np.unique(events['session']):
        sess_data = events[events['session']==sess]
        global_pos_rate = np.mean(sess_data['recalled'])
        for sp in np.unique(sess_data['serialpos']):
            sp_pos_rate = np.mean(sess_data[(sess_data['serialpos']==sp)]['recalled'])
            pos_data = sess_data[(sess_data['serialpos']==sp)&(sess_data['recalled']==1)]
            neg_data = sess_data[(sess_data['serialpos']==sp)&(sess_data['recalled']==0)]
            if sp_pos_rate >= global_pos_rate:
                # more pos data from this sp than global avg
                # keep all neg data and downsample pos data
                n = np.round(len(neg_data)/(1-global_pos_rate)) - len(neg_data)
                resampled_events_mstime.extend(np.random.choice(pos_data['mstime'], size=int(n), replace=False))
                resampled_events_mstime.extend(neg_data['mstime'])
            else: # if sp_pos_rate < global_pos_rate:
                # less pos data from this sp than global avg
                # keep all pos data and downsample neg data
                resampled_events_mstime.extend(pos_data['mstime'])
                n = np.round((len(pos_data)/global_pos_rate)) - len(pos_data)
                resampled_events_mstime.extend(np.random.choice(neg_data['mstime'], size=int(n), replace=False))

    feature = feature.sel(events=np.in1d(events['mstime'], resampled_events_mstime))
    return feature

def resample_ret_events(feature):
    rectime_bins = np.arange(0, 75.1, 7.5)*1000

    events = feature.events.values.copy()
    resampled_events_mstime = []
    for sess in np.unique(events['session']):
        sess_data = events[events['session']==sess]
        global_pos_rate = np.mean(sess_data['rec_label']==1)
        for b in range(1, 11):
            tmin = rectime_bins[b-1]
            tmax = rectime_bins[b]        
            bin_data = sess_data[(sess_data['rectime']>tmin)&(sess_data['rectime']<=tmax)]
            if len(bin_data)==0:
                continue
            bin_pos_rate = np.mean(bin_data['rec_label']==1)
            pos_data = bin_data[bin_data['rec_label']==1]
            neg_data = bin_data[bin_data['rec_label']==0]
            if bin_pos_rate >= global_pos_rate:
                # more pos data from this bin than global avg
                # keep all neg data and downsample pos data
                if len(neg_data)==0:
                    continue
                n = np.round(len(neg_data)/(1-global_pos_rate)) - len(neg_data)
                resampled_events_mstime.extend(np.random.choice(pos_data['mstime'], size=min(len(pos_data), int(n)), replace=False))
                resampled_events_mstime.extend(neg_data['mstime'])
            else:
                # less pos data from this bin than global avg
                # keep all pos data and downsample neg data
                if len(pos_data)==0:
                    continue
                resampled_events_mstime.extend(pos_data['mstime'])
                n = np.round((len(pos_data)/global_pos_rate)) - len(pos_data)
                resampled_events_mstime.extend(np.random.choice(neg_data['mstime'], size=min(len(neg_data),int(n)), replace=False))

    feature = feature.sel(events=np.in1d(events['mstime'], resampled_events_mstime))
    return feature

def split_ROIs(power, sys):

    # by eight rois
    ROI_list = ['LAS','LAI','LPS','LPI','RAS','RAI','RPS','RPI']

    power_byroi = []
    for roi in ROI_list:
        channels = ROIs[sys][roi]
        x = power.sel(channels=np.in1d(power.channels.values, channels)).mean('channels')
        power_byroi.append(x)

    power_byroi = xr.concat(power_byroi, dim='ROI')
    power_byroi.coords['ROI'] = ROI_list

    return power_byroi

def special_concat(arr_of_ts):
    '''
    Can be used to concatenate timeseries object that contains the CML events dimension
    To get around the problem when PTSA concatenates multiple timeseries object the events dim
    will turn out to have dype='O'
    Assumes the timeseries objects to have a 'events' dim
    '''
    if len(arr_of_ts)==1:
        return arr_of_ts[0]
    else:
        x = xr.concat(arr_of_ts, dim='events')
        events = arr_of_ts[0].events.values
        for ts in arr_of_ts[1:]:
            events = np.concatenate((events, ts.events.values))
        x.coords['events'] = events
        return x

def divide_folds(sessions, nfold):
    '''
    creates the cross-validation assignment (of session numbers)
    '''
    n = int(np.floor(len(sessions) / nfold))
    folds = []
    for i in range(0, len(sessions), n):
        folds.append(sessions[i:i+n])
    return np.array(folds)

def compute_activation(X, weights):
    '''A = cov(X) * W / cov(y_hat)'''

    activations = np.cov(X.T).dot(weights) / np.cov(X.dot(weights))

    # activations shape: N features array
    return activations

def leave_one_session_out_forward_model(train_features, train_label, test_features, test_label, C, regularization='l2'):
        
    probas = np.empty(len(test_features)) # however many events in data
    labels = np.empty(len(test_features))
    probas.fill(np.nan)
    labels.fill(np.nan)

    weights = []
    activations = []

    folds = divide_folds(np.unique(train_features.events.values['session']),
                         len(np.unique(train_features.events.values['session']))) # LOSO

    for fold in folds:

        train = np.array([False if s in fold else True for s in train_features.events.values['session']])
        test = np.array([True if s in fold else False for s in test_features.events.values['session']])

        train_data = train_features.sel(events=train).copy()
        test_data = test_features.sel(events=test).copy()

        X = train_data.values
        y = train_data.events.values[train_label]

        if regularization=='l2':
            lr = LogisticRegression(C=C, class_weight='balanced', fit_intercept=False)
        elif regularization=='l1':
            lr = LogisticRegression(C=C, class_weight='balanced', fit_intercept=False,
                                   penalty='l1', solver='liblinear')
        lr = lr.fit(X, y)

        # weights and activation values from the corresponding forward model
        weights.append(lr.coef_.flatten()) # flatten from 1*Nfeatures to Nfeatures

        # find out how much each spectral feature contributed to the classification
        A = compute_activation(X, lr.coef_.flatten())
        activations.append(A.flatten())

        probas[test] = lr.predict_proba(test_data)[:, 1]
        labels[test] = test_data.events.values[test_label]

    return (probas, labels, np.array(weights), np.array(activations))

def shuffle_labels(events, label):
    shuffled_labels = np.zeros(len(events)).astype(int)
    for s in np.unique(events['session']):
        sess_index = events['session']==s
        labels = events[sess_index][label]
        np.random.shuffle(labels)
        shuffled_labels[sess_index] = labels

    events[label] = shuffled_labels
    return events

def compute_ROC_AUC(y, y_hat, base=200):
    base = np.linspace(1, 0, base) # make 200 points
    fp, tp, t = roc_curve(y, y_hat)
    s = np.argsort(fp) # fp (x-axis) must be increasing for scipy.interp
    interp = scipy.interp(base, fp[s], tp[s])
    aucscore = auc(fp, tp)

    return interp, aucscore

def normalize_by_pop_mean(x_in):
    # x is subject by condition
    x = x_in.copy()
    grand_mean = np.nanmean(np.nanmean(x,1))
    for i, row in enumerate(x):
        dev = grand_mean - np.nanmean(row)
        x[i] = row + dev
    return x

def get_behavioral_score(subject):
    was_sims = np.loadtxt('/home1/liyuxuan/notebooks/pools/wasnorm_was.txt')
    
    def get_subject_prec(recalls):
        return np.mean(np.mean(recalls, 1))
    
    def get_subject_primacy(recalls):
        early = np.mean(np.mean(recalls[:,:4], 0))
        middle = np.mean(np.mean(recalls[:,4:-4], 0))
        return early-middle
    
    with open('/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_%s.json'%subject) as f:
        x = js.load(f)
        p = get_subject_prec(np.array(x['recalled'])[np.array(x['good_trial'])])
        pf = get_subject_primacy(np.array(x['recalled'])[np.array(x['good_trial'])])
        tf = temp_fact(np.array(x['serialpos'])[np.array(x['good_trial'])],
                       np.array(x['subject'])[np.array(x['good_trial'])],
                       listLength = 24)
        sf = dist_fact(np.array(x['rec_nos'])[np.array(x['good_trial'])],
                       np.array(x['pres_nos'])[np.array(x['good_trial'])],
                       np.array(x['subject'])[np.array(x['good_trial'])],
                       dist_mat = was_sims,
                       is_similarity = True)
    
    return {'prec':p, 'primacy':pf, 'temp':tf, 'sem':sf}