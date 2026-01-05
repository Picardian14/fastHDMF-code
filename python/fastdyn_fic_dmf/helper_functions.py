import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.stats import zscore

def filter_bold(bold, flp, fhp, tr):
    
    T, N = bold.shape
    fnq = 1 / (2 * tr)  # Nyquist frequency
    Wn = [flp / fnq, fhp / fnq]  # Butterworth bandpass non-dimensional frequency
    k = 2  # 2nd order Butterworth filter
    bfilt, afilt = butter(k, Wn, btype='band')  # Construct the filter

    # Filtering and plotting
    filt_bold = np.zeros((T, N))
    nzeros = 40
    aux_filt = detrend(bold, axis=0)
    aux_filt = np.concatenate((np.zeros((nzeros, N)), aux_filt, np.zeros((nzeros, N))))

    for n in range(N):
        aux_filt2 = filtfilt(bfilt, afilt, aux_filt[:, n])  # Zero-phase filter the data
        filt_bold[:, n] = zscore(aux_filt2[nzeros:-nzeros])

    return filt_bold

def compute_fcd(data, wsize, overlap, isubdiag):
    T, N = data.shape
    win_start = np.arange(0, T - wsize - 1, wsize - overlap)
    nwins = len(win_start)
    fcd = np.zeros((len(isubdiag[0]), nwins))
    #print(fcd.shape)
    #print(data.shape)
    #print((data[win_start[2]:win_start[2] + wsize + 1, :]).shape)
    for i in range(nwins):
        tmp = data[win_start[i]:win_start[i] + wsize + 1, :]
        cormat = np.corrcoef(tmp.T)
        fcd[:, i] = cormat[isubdiag[0],isubdiag[1]]
    return fcd