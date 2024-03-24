import numpy as np
from scipy.signal import find_peaks
from biosppy.utils import ReturnTuple
from scipy.stats import variation
from biosppy import tools as st


def longest_consecutive_ones(bits01):
    """
    Get the length of the longest consecutive 1s in a binary array.

    :param bits01: Binary array, e.g., np.array[0, 0, 1, 1, 1]
    :return: Length of the longest consecutive 1s
    """
    bits_n1_p1 = bits01 * 2 - 1  # e.g. array[-1, -1, 1, 1, 1]
    run_starts = np.where(np.diff(np.hstack(([0], bits_n1_p1))) > 0)[0]
    run_ends = np.where(np.diff(np.hstack((bits_n1_p1, [0]))) < 0)[0]
    return max(run_ends - run_starts + 1)


def get_troughs(ecg_sig, r_loc, ecg_fs, tol=0.1):
    """
    Find troughs in an interval before the R peak locations for each R peak.

    :param ecg_sig: Input ECG signal array
    :param r_loc: R peak locations array
    :param ecg_fs: Sampling rate of the ECG signal
    :param tol: Interval in seconds to search for local minima (default: 0.1)
    :return: Locations of troughs for each R peak
    """
    q_loc = np.zeros(len(r_loc), dtype=np.int)
    for i, r in enumerate(r_loc):
        temp_start = max(0, r - int(tol * ecg_fs))
        if temp_start < r:
            q_loc[i] = np.argmin(ecg_sig[temp_start:r]) + temp_start
        else:
            q_loc[i] = r
    return q_loc


def get_qr_peaks(ecg_sig, ecg_fs):
    """
    Get R peaks and troughs from the ECG signal.

    :param ecg_sig: Input ECG signal array
    :param ecg_fs: Sampling rate of the ECG signal
    :return: R peak locations and trough locations
    """
    signal = np.array(ecg_sig)
    order = int(0.3 * ecg_fs)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=ecg_fs)
    r_loc, = hamilton_segmenter(signal=filtered, sampling_rate=ecg_fs)
    r_loc, = correct_rpeaks(signal=filtered,
                            rpeaks=r_loc,
                            sampling_rate=ecg_fs,
                            tol=0.05)
    q_loc = get_troughs(ecg_sig, r_loc, ecg_fs)
    return r_loc, q_loc


def _elim_outlier(estimates, qualities=None):
    """
    Eliminate outliers from the estimates based on a threshold.

    :param estimates: Array of estimates
    :param qualities: Array of qualities (default: None)
    :return: Filtered estimates and qualities (if provided)
    """
    estimates = np.array(estimates)

    selected_idx = estimates < 60
    if qualities is not None:
        qualities = np.array(qualities)
        return estimates[selected_idx], qualities[selected_idx]
    return estimates[selected_idx]


def smart_fusion(estimates, sd_thresh=4):
    """
    Perform smart fusion of estimates based on the variation of estimates from different sources.

    :param estimates: Array of estimated respiratory rates
    :param sd_thresh: Maximum standard deviation allowed (default: 4.0)
    :return: Fused estimate (0 if variation is too big)
    """
    estimates = np.array(estimates)
    estimates = estimates[estimates > 0]
    if len(estimates) == 0 or np.std(estimates) > sd_thresh:
        return 0
    estimates = _elim_outlier(estimates)
    return np.mean(estimates)


def quality_fusion(estimates, qualities, sd_thresh=0):
    """
    Combine estimates by calculating the average weighted by the signal qualities.

    :param estimates: Array of estimates
    :param qualities: Array of qualities
    :param sd_thresh: If > 0, the estimates will first be checked for variations (default: 0)
    :return: Fused estimate
    """
    assert len(estimates) == len(qualities)
    estimates, qualities = _elim_outlier(estimates, qualities)
    if 0 < sd_thresh < np.std(estimates) or sum(qualities) <= 0:
        return 0
    return np.average(estimates, weights=qualities)


def detect_burst_noise(ecg_buffer, burst_thres):
    """
    Detect whether the signal contains burst noise using the burst noise threshold.

    :param ecg_buffer: Raw ECG signal array
    :param burst_thres: Burst noise threshold
    :return: 1 if burst noise is present, 0 otherwise
    """
    if abs(max(ecg_buffer) - min(ecg_buffer)) > burst_thres:
        ecg_is_noise = 1
    else:
        ecg_is_noise = 0
    return ecg_is_noise


def detect_noinput(ecg_buffer, SAMPLING_RATE, timeinsec=5, noinput_thr=0.05):
    """
    Detect whether the signal contains segments with missing values.

    :param ecg_buffer: Raw ECG signal array
    :param SAMPLING_RATE: Sampling frequency (Hz)
    :param timeinsec: Cutoff for time in seconds (default: 5)
    :param noinput_thr: Amplitude cutoff for missing data (default: 0.05)
    :return: 1 if missing values are present, 0 otherwise
    """
    total_noinput_samples = SAMPLING_RATE * timeinsec  # len(strip)
    a_noinput = abs(ecg_buffer) < noinput_thr
    ecg_nosig_ind = a_noinput * 1  # convert to integer
    try:
        longest_1s = longest_consecutive_ones(ecg_nosig_ind)
        if longest_1s > total_noinput_samples:
            ecg_is_noinput = 1
        else:
            ecg_is_noinput = 0
    except:
        ecg_is_noinput = 1

    return ecg_is_noinput


def quality_metric(signal=None):
    """
    Extract the quality metric of the respiration signal.

    :param signal: Raw respiration signal array
    :return: Quality metric denoting the quality of the respiration signal (0-1)
    """
    if signal is None:
        q_metric = 0
    else:
        signal = np.array(signal)
        extrema, values = st.find_extrema(signal, mode="max")
        cv_extrema = np.abs(variation(np.diff(extrema)))
        cv_value = np.abs(variation(values))
        q_metric = np.round(np.mean([math.exp(-cv_extrema), math.exp(-cv_value)]), 2)

    args = (q_metric, )
    names = ('quality', )
    return ReturnTuple(args, names)


def chan_quality(signal, peaks, troughs, num_maxima, num_minima):
    """
    Calculate the quality of the respiration signal based on the Chan et al. method.

    :param signal: Respiratory signal array
    :param peaks: Array of peak locations
    :param troughs: Array of trough locations
    :param num_maxima: Total number of local maxima in the signal
    :param num_minima: Total number of local minima in the signal
    :return: Quality metric and related parameters
    """
    if len(peaks) <= 2:
        quality = 0
        cov_pp_amp = mean_pp_amp = cov_min_int = cov_max_int = extrema_ratio = 0
    else:
        cov_pp_amp = variation(signal[peaks] - signal[troughs]) + 1e-4
        mean_pp_amp = np.mean(signal[peaks] - signal[troughs])
        cov_min_int = variation(troughs[1:] - troughs[:-1]) + 1e-4
        cov_max_int = variation(peaks[1:] - peaks[:-1]) + 1e-4
        extrema_ratio = (len(peaks) + len(troughs)) / (num_maxima + num_minima)
        quality = 0.5*(1 / cov_max_int + 1 / cov_min_int) / cov_pp_amp * extrema_ratio

    args = (quality, cov_pp_amp, mean_pp_amp, cov_min_int, cov_max_int, extrema_ratio)
    names = ('quality', 'cov_pp_amp', 'mean_pp_amp', 'cov_min_int', 'cov_max_int', 'extrema_ratio')
    return ReturnTuple(args, names)


def interval_quality(signal, intervals, peaks, var_thresh=0.5):
    """
    Assess the quality of intervals between peaks of the respiratory signal.

    :param signal: Respiratory signal array
    :param intervals: Array of identified intervals
    :param peaks: Array of peak locations associated with the intervals
    :param var_thresh: Threshold of maximum variation coefficient allowed (default: 0.5)
    :return: Quality metric
    """
    if len(intervals) == 0:
        quality = 0.
    else:
        x_var = variation(intervals)
        x_len = len(intervals)
        cov_pp_amp = variation(signal[peaks])
        quality = (x_var < var_thresh) * (1/(x_var*cov_pp_amp+1e-4))**2 * np.log2(x_len)
    args = (quality,)
    names = ('quality',)

    return ReturnTuple(args, names)


def resampling(timestamp, data, resampling_rate, cubic=True):
    """
    Resample the time series and signal using linear or cubic spline interpolation.

    :param timestamp: Timestamp array of the input signal
    :param data: Non-uniformly sampled input signal array
    :param resampling_rate: Resampling frequency (Hz)
    :param cubic: Whether to use cubic spline interpolation (default: True)
    :return: Resampled time series and signal
    """
    if cubic:
        interp = CubicSpline(timestamp, data)
    else:
        interp = interp1d(timestamp, data)

    resampled_time = np.arange(timestamp[0] + 1/resampling_rate, timestamp[-1], 1 / resampling_rate)
    resampled_data = interp(resampled_time)
    return resampled_time, resampled_data


def plotting(rr_metrics):
    """
    Plot the respiration rates and quality metrics of the modulations.

    :param rr_metrics: DataFrame containing metrics estimated using ECG-derived respiration
    :return: Plot containing respiration rate and quality metrics
    """
    fig, axes = plt.subplots(nrows=2, ncols=1)
    if rr_metrics.shape[0] > 0:
        rr_metrics[[x for x in rr_metrics.columns if x.startswith('RR_')]].plot(linestyle="-", ax=axes[0])
    axes[0].set_ylim([6, 60])
    axes[0].set_ylabel("Respiration Rate")
    axes[0].set_xticks([])

    if rr_metrics.shape[0] > 0:
        rr_metrics[[x for x in rr_metrics.columns if x.startswith('Q_')]].plot(linestyle="-", ax=axes[1])

    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("Quality Metric")

    plt.legend()
    return plt


def count_orig(signal, percent=75, height_prop=0.2):
    """
    Find peaks in the respiration signal using the Count_orig method.

    :param signal: Respiration signal array
    :param percent: Percentile of peak heights used for filtering the selected peaks (default: 75)
    :param height_prop: Proportion of the percentile height to used as filter (default: 0.2)
    :return: Intervals, selected peaks, and selected troughs
    """
    if height_prop is None:
        height_prop = 0.2
    signal = np.array(signal)
    all_peaks, _ = find_peaks(signal)
    all_mins, _ = find_peaks(-signal)
    if len(all_peaks) > 0 and len(all_mins) > 0:
        thresh = height_prop * np.percentile(signal[all_peaks], percent)
        peaks = all_peaks[signal[all_peaks] > thresh]
        troughs = all_mins[signal[all_mins] < 0]
        min_counts = np.zeros(len(peaks) - 1)
        min_idxes = []

        if len(peaks) < 1:
            intervals = []
            selected_peaks = []
            selected_troughs = []
        else:
            for i in range(len(peaks)-1):
                min_idxes.append((peaks[i] < troughs) & (troughs < peaks[i+1]))
                min_counts[i] = np.sum(min_idxes[-1])
            intervals = np.diff(peaks)[min_counts == 1]
            selected_peaks = [peaks[i] for i in range(len(peaks))
                              if (i < len(min_counts) and min_counts[i] == 1)
                              or (i - 1 >= 0 and min_counts[i-1] == 1)]
            selected_troughs = [troughs[min_idx][0] for min_idx in min_idxes if np.sum(min_idx) == 1]
    else:
        intervals = []
        selected_peaks = []
        selected_troughs = []
        peaks = []
        troughs = []

    args = (intervals, selected_peaks, selected_troughs, len(peaks), len(troughs))
    names = ('intervals', 'peaks', 'troughs', 'num_max', 'num_min')
    return ReturnTuple(args, names)


def chan_peak(signal, sampling_rate, alpha=0.7, num_seconds=3, percent=75, height_prop=None):
    """
    Find peaks in the respiration signal using the Chan et al. method.

    :param signal: Respiration signal array
    :param sampling_rate: Sampling rate of the signal
    :param alpha: Threshold (default: 0.7)
    :param num_seconds: Number of seconds to calculate standard deviation (default: 3)
    :param percent: Percentile of peak heights used for filtering the selected peaks (default: 75)
    :param height_prop: Proportion of the percentile height to used as filter (default: None)
    :return: Intervals, selected peaks, selected troughs, and other related parameters
    """
    signal = np.array(signal)
    interval = int(np.ceil(sampling_rate * num_seconds))

    maxima_loc = find_peaks(signal)[0]
    minima_loc = find_peaks(-signal)[0]
    maxima_val = signal[maxima_loc]
    minima_val = signal[minima_loc]
    can_min = 0
    can_max = -1
    selected_maxs = []
    selected_mins = []
    threshes = []
    i = 1
    j = 0

    # Make sure the first start with local minima
    while j < len(maxima_loc) and i < len(minima_loc) and minima_loc[0] > maxima_loc[j]:
        j += 1

    while i < len(minima_loc) and j < len(maxima_loc):
        if minima_loc[i] < maxima_loc[j]:
            thresh = alpha * np.std(signal[max(0, minima_loc[i] - interval):(minima_loc[i] + 1)])
            if can_max >= 0:
                if maxima_val[can_max] - minima_val[i] > thresh:
                    selected_maxs.append(can_max)
                    selected_mins.append(can_min)
                    threshes.append(thresh)
                    can_min = i
                    can_max = -1
                elif minima_val[i] < minima_val[can_min]:
                    can_min = i
            else:
                if minima_val[i] < minima_val[can_min]:
                    can_min = i
            i += 1
        else:
            thresh = alpha * np.std(signal[max(0, maxima_loc[j] - interval):(maxima_loc[j] + 1)])
            if maxima_val[j] - minima_val[can_min] > thresh:
                if can_max < 0 or maxima_val[j] > maxima_val[can_max]:
                    can_max = j
            j += 1

    if i < len(minima_loc) and can_max >= 0:
        thresh = alpha * np.std(signal[max(0, minima_loc[i] - interval):(minima_loc[i] + 1)])
        if maxima_val[can_max] - minima_val[i] > thresh:
            selected_maxs.append(can_max)
            selected_mins.append(can_min)
            threshes.append(thresh)

    selected_maxs = np.array(selected_maxs)
    selected_mins = np.array(selected_mins)
    threshes = np.array(threshes)

    if len(selected_maxs) < 1:
        intervals = np.array([])
        peaks = np.array([])
        troughs = np.array([])
    else:
        peaks = maxima_loc[selected_maxs]
        troughs = minima_loc[selected_mins]
        if (len(peaks) > 2) and (height_prop is not None):
            percent_height = np.percentile(signal[peaks], percent)
            if isinstance(height_prop, list):
                temp_prop, max_prop = height_prop[:2]
                all_variations = [np.inf]
                all_idxes = []
                while temp_prop <= max_prop:
                    thresh = temp_prop * percent_height
                    temp_idx = (signal[peaks] > thresh) & (signal[troughs] < 0)
                    all_idxes.append(temp_idx)
                    temp_peaks = peaks[temp_idx]
                    if len(temp_peaks) > 2:
                        all_variations.append(variation(temp_peaks[1:] - temp_peaks[:-1]))
                    else:
                        all_variations.append(np.nan)
                    temp_prop += 0.1
                if np.nanargmin(all_variations) > 0:
                    selected_idx = all_idxes[np.nanargmin(all_variations) - 1]
                else:
                    selected_idx = None
            else:
                thresh = height_prop * percent_height
                selected_idx = (signal[peaks] > thresh) & (signal[troughs] < 0)
            if selected_idx is not None:
                troughs = troughs[selected_idx]
                threshes = threshes[selected_idx]
                peaks = peaks[selected_idx]
            else:
                troughs = []
                threshes = []
                peaks = []

        if len(peaks) > 1:
            intervals = peaks[1:] - peaks[:-1]
        else:
            intervals = np.array([])

    args = (intervals, peaks, troughs, len(maxima_loc), len(minima_loc), threshes)
    names = ('intervals', 'peaks', 'troughs', 'num_max', 'num_min', 'threshes')
    return ReturnTuple(args, names)

def zero_crossing(signal):
    """
    Find intervals and peaks in the respiration signal using the zero-crossing method.

    :param signal: Respiration signal array
    :return: Tuple containing intervals and peaks
    """
    intervals = []
    peaks = []

    if len(signal) > 0:
        zeros, = st.zero_cross(signal=signal, detrend=True)
        peaks = zeros[::2]

        if len(peaks) >= 2:
            intervals = np.diff(peaks)
        else:
            intervals = []
            peaks = []

    args = (intervals, peaks)
    names = ('intervals', 'peaks')
    return ReturnTuple(args, names)
