import numpy as np
import mne
import scipy
import xarray as xr
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import normalize_by_pop_mean

def plot_heatmap_contrast(list_of_values, vmin, vmax, figsize, colorbar=True, fdr_correction=True):
    
    '''
    plots a heatmap with significant group-test regions preserved and insignificant ones marked out
    
    :param list_of_values: must be 2d arrays
    :param vmin, vmax: ends of color scale
    
    '''
    
    def hide_spines_and_ticks(ax):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both',length=0)
    
    def get_FDR_sigmatrix(X):
        # defaults to Benjamini/Hochberg method in mne
        tvals, pvals = scipy.stats.ttest_1samp(X, 0, axis=0, nan_policy='omit')
        reject, pval_corrected = mne.stats.fdr_correction(pvals.flatten())
        sigmatrix = reject.reshape(pvals.shape)
        return tvals, sigmatrix
    
    def imshow_wrapper(ax, valuematrix, sigmatrix, vmin, vmax, cmap):

        '''plots time-frequency spectrogram and marks significance, but axis and ticks are handled outside'''
        
        if fdr_correction:
            sig_region = np.ma.masked_where(~sigmatrix, valuematrix)
            nonsig_region = np.ma.masked_where(sigmatrix, valuematrix)
            ax.imshow(sig_region, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.imshow(nonsig_region, origin='lower', aspect='auto', cmap=cmap, alpha=1, vmin=-100000, vmax=100000)
        else:
            ax.imshow(valuematrix, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        hide_spines_and_ticks(ax)
        
        return ax
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if colorbar:
        cbar_ax = fig.add_axes([.88, .3, .03, .4])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap='RdBu_r', norm=norm, orientation='vertical')
        hide_spines_and_ticks(cb1.ax)
        cb1.outline.set_visible(False)
        cb1.ax.text(0, 0, 't', ha='center', va='center')
        # cb1.ax.text(0.5, 0.5, 't', ha='center', va='center')
    
    # concat along subject dim
    X = xr.concat(list_of_values, dim='subjects')
    tvals, sig_matrix = get_FDR_sigmatrix(X.values)
    # plot average relative t-test tstat
    ax = imshow_wrapper(ax, X.mean('subjects').T, sig_matrix.T, vmin=vmin, vmax=vmax, cmap='RdBu_r')
    # plot one-sample t-test tstat
    # ax = imshow_wrapper(ax, tvals.T, sig_matrix.T, vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
    freqs = list_of_values[0].frequency.values
    ax.set_yticks(range(len(freqs))[::4])
    ax.set_yticklabels(['%d'%f for f in freqs[::4]])
    ax.set_xticks(range(8))
    ax.set_xticklabels(['LAI','LAS','LPS','LPI','RAI','RAS','RPS','RPI'])
    ax.set_ylabel('Frequency (Hz)', size=12)
    ax.set_xlabel('ROIs', size=12)

    fig.tight_layout(rect=[0,0,0.89,1], h_pad=0.2)
    return fig, ax

def plot_example_lists(lists, resample=False, figsize=[6.5,4]):
    
    def get_example_trial(subject, sess, trial):
        if not resample:
            events_key = 'events'
            enc = pkl.load(open('scratch/mtpower/%s/%s_enc_enc_clsf_result.pkl'%(subject, subject), 'rb'))
        if resample:
            events_key = 'test_events'
            enc = pkl.load(open('scratch/mtpower/%s/%s_enc_enc_clsf_resample_run1_result.pkl'%(subject, subject), 'rb'))
        ency = enc['y'][(enc[events_key]['session']==sess) & (enc[events_key]['trial']==trial)]
        encyhat = enc['yhat'][(enc[events_key]['session']==sess) & (enc[events_key]['trial']==trial)]

        if not resample:
            events_key = 'events'
            ret = pkl.load(open('scratch/mtpower/%s/%s_retci_retci_clsf_result.pkl'%(subject, subject), 'rb'))
        if resample:
            events_key = 'test_events'
            ret = pkl.load(open('scratch/mtpower/%s/%s_retci_retci_clsf_resample_run1_result.pkl'%(subject, subject), 'rb'))
        rety = ret['y'][(ret[events_key]['session']==sess) & (ret[events_key]['trial']==trial)]
        retyhat = ret['yhat'][(ret[events_key]['session']==sess) & (ret[events_key]['trial']==trial)]
        rectimes = ret[events_key]['rectime'][(ret[events_key]['session']==sess) & (ret[events_key]['trial']==trial)]

        return (ency, encyhat, rety, retyhat, rectimes)

    def find_bin(t, bins=np.arange(0, 75.1, 7.5)):
        return np.where(t>bins)[0][-1]

    def plot_encoding(ax, ency, encyhat):
        for sp in range(24):
            ax.scatter(sp+1, encyhat[sp], color='k' if ency[sp]==1 else 'k', 
                       marker='o' if ency[sp]==1 else 'x', 
                       facecolor='w' if ency[sp]==1 else 'k', s=20)
        ax.set_xticks(np.arange(1,25,1)[1::4])
        ax.set_xlim([0,25])
        ax.set_yticks([0.3,0.5,0.7])
        ax.set_yticklabels(['0.3','0.5','0.7'], fontsize=9)

    def plot_retrieval(ax, retinty, retintyhat, rectimes):

        for i, t in enumerate(rectimes):
            ax.scatter(t/1000.0, retintyhat[i], color='k' if retinty[i]==1 else 'k',
                       marker='o' if retinty[i]==1 else 'x',
                       facecolor='w' if retinty[i]==1 else 'k', s=20)
        ax.set_xticks(np.arange(0, 75.1, 7.5)[::2])
        ax.set_xlim([0,75])
        ax.set_yticks([0.3,0.5,0.7])
        ax.set_yticklabels(['0.3','0.5','0.7'], fontsize=9)
    
    fig, axes = plt.subplots(len(lists),2, figsize=figsize, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0.5, lw=1, ls='--', color='k', alpha=0.5)
        if i%2!=0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)
        if i<4:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', bottom=False)
            ax.set_xticklabels('')
        # plot axis breaks
        dx = 0.3
        dy = 0.05
        if i%2==0:
            ax.plot([25, 25],[0.5-dy, 0.5+dy], 'k', lw=1, alpha=0.6, clip_on=False)
        if i%2!=0:
            ax.plot([dx, dx],[0.5-dy, 0.5+dy], 'k', lw=1, alpha=0.6, clip_on=False)

    for i, (subject, sess, trial) in enumerate(lists):
        data = get_example_trial(subject, sess, trial)
        ax = axes[i][0]
        plot_encoding(ax, data[0], data[1])
        ax = axes[i][1]
        plot_retrieval(ax, data[2], data[3], data[4])

    # these are hard-coded for 3 subjects...
    axes[2,0].set_xticklabels(np.arange(1,25,1)[1::4], fontsize=9)
    axes[2,1].set_xticklabels([int(x) for x in np.arange(0, 75.1, 7.5)[::2]], fontsize=9)

    axes[0,0].set_title('Encoding', fontsize=10)
    axes[0,1].set_title('Retrieval', fontsize=10)
    axes[2,0].set_xlabel('Serial Position', fontsize=9)
    axes[2,1].set_xlabel('Recall Time (s)', fontsize=9)
    axes[1,0].set_ylabel('Classifier Output', fontsize=9)

    fig.tight_layout(h_pad=0, w_pad=0.6)
    
    return fig, axes

def plot_ROC(ax, roc, text=None):
    ax.plot(np.linspace(1, 0, len(roc)), roc, color='k')
    
    ax.set_ylim(-0.02,1.02)
    ax.set_xlim(-0.02,1.02)
    ax.set_yticks([0,1])
    ax.set_xticks([0,1])

    midline = [0,0.2,0.4,0.6,0.8,1.0]
    ax.plot(midline, midline, '--k', alpha=0.5, lw=1)
    
    if text is not None:
        ax.text(0, 0.7, text, ha='left')
    
    return ax

def plot_auc_null(observed_auc, null_aucs, figsize=[2,1.2]):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(null_aucs, color='w', edgecolor='gray', bins=np.arange(0.35,0.76,0.01))
    ax.hist(observed_auc, color='k', edgecolor='k', bins=np.arange(0.35,0.76,0.01), alpha=0.52)
    ax.set_xlim([0.35, 0.75])
    ax.set_xticks([0.4,0.5,0.6,0.7])
#     ax.set_ylim([0, 30])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax

def plot_list_dynamics(value_dict, figsize=[5,3.5], normalize_by_pop_means=True):
    
    '''
    :param value_dict: a dict with event type as keys and value array and aesthetics (e.g., label, color, marker) as values
    '''
    
    # output by clustering type and serialpos/recall bin
    fig, ax = plt.subplots(figsize=figsize)
    
    for event_type in value_dict.keys():
        data = value_dict[event_type]
        if normalize_by_pop_means:
            x = normalize_by_pop_mean(data['value'])
        
        ax.plot(data['xticks']+data['jitter'], np.nanmean(x, 0), 
                ls=data['linestyle'], marker=data['marker'], ms=data['markersize'], mfc='w',
                color=data['color'], label=event_type)
        err = scipy.stats.sem(x, axis=0, nan_policy='omit')
        ax.fill_between(data['xticks']+data['jitter'], np.nanmean(x, 0)-err, np.nanmean(x, 0)+err, color='k', alpha=0.07, edgecolor='w')
        
    fig.tight_layout()
    return fig, ax