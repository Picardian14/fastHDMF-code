import numpy as np
from scipy.signal import butter, welch, detrend, filtfilt
from scipy.stats import zscore


def filter_bold_fft(bold: np.ndarray, flp: float, fhp: float, tr: float) -> np.ndarray:
    """
    Apply a bandpass filter to the BOLD signal using frequency domain filtering with Welch method concepts.
    Now expects bold shape to be (N, T) where N is regions and T is time points.
    """
    N, T = bold.shape  # Changed from T, N to N, T
    fnq = 1 / (2 * tr)  # Nyquist frequency
    
    # Detrend the data along time axis (axis=1 now)
    detrended_bold = detrend(bold, axis=1)
    
    # Initialize output
    filt_bold = np.zeros((N, T))
    
    # Create frequency array
    freqs = np.fft.fftfreq(T, tr)
    
    # Create bandpass filter mask
    freq_mask = (np.abs(freqs) >= flp) & (np.abs(freqs) <= fhp)
    
    # Apply filtering in frequency domain for each region
    for n in range(N):
        # FFT of the signal
        signal_fft = np.fft.fft(detrended_bold[n, :])
        
        # Apply bandpass filter
        filtered_fft = signal_fft * freq_mask
        
        # Inverse FFT to get filtered signal
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        # Z-score normalize
        filt_bold[n, :] = zscore(filtered_signal)
    
    return filt_bold

def filter_bold(bold: np.ndarray, flp: float, fhp: float, tr: float) -> np.ndarray:
    N, T = bold.shape
    fnq = 1 / (2 * tr)  # Nyquist frequency
    Wn = [flp / fnq, fhp / fnq]  # Non-dimensional frequency for the Butterworth filter
    k = 2  # 2nd order Butterworth filter
    bfilt, afilt = butter(k, Wn, btype='band')  # Construct the filter

    # Filtering the signal
    filt_bold = np.zeros((N, T))
    nzeros = 40
    # Detrend the data along the time axis (axis=1) and pad zeros at beginning and end along time axis
    aux_filt = detrend(bold, axis=1)
    aux_filt = np.concatenate((np.zeros((N, nzeros)), aux_filt, np.zeros((N, nzeros))), axis=1)

    for n in range(N):
        aux_filt2 = filtfilt(bfilt, afilt, aux_filt[n, :])  # Zero-phase filter the data for each region
        filt_bold[n, :] = zscore(aux_filt2[nzeros:-nzeros])  # Remove padding and z-score normalize

    return filt_bold


def compute_fcd(data: np.ndarray, wsize: int, overlap: int, isubdiag: tuple) -> np.ndarray:
    N, T = data.shape
    win_start = np.arange(0, T - wsize - 1, wsize - overlap)
    nwins = len(win_start)
    fcd = np.zeros((len(isubdiag[0]), nwins))
    for i in range(nwins):
        tmp = data[:,win_start[i]:win_start[i] + wsize + 1]
        cormat = np.corrcoef(tmp)
        fcd[:, i] = cormat[isubdiag[0], isubdiag[1]]
    return fcd


def to_signal_percentage(data: np.ndarray) -> np.ndarray:
    """
    Convert the input data to a percentage of the maximum value for each time series.
    """
    #TODO
    max_vals = np.max(data, axis=0, keepdims=True)
    return data / max_vals * 100