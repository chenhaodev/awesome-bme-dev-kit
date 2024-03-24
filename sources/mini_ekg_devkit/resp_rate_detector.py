import numpy as np
from biosppy.signals import tools as st
from scipy.signal import detrend
import resampy
from scipy.stats import trim_mean

from resp_rate_util import fix_baseline_wander, filter_resp_signal, chan_peak, count_orig, zero_crossing, chan_quality, interval_quality, get_qr_peaks, quality_fusion, smart_fusion

class RespRateEstimator:
    def __init__(self):
        """
        Initialize the RespRateEstimator.
        """
        self.ecg_fs = None
        self.ecg_raw = None

        self.burst_noise_thr = None
        self.signals_to_use = None
        self.resp_fs = None
        self.use_wavelet = None
        self.peak_method = None
        self.fusion_method = None
        self.trim_mean_cutoff = None
        self.percentile = None
        self.height_prop = None
        self.sd_thresh = None
        self.kernel_size = None
        self.stride_length = None

        self.classifier_output = None
        self.ectopic_beats = None
        self.ectopic_intervals = None
        self.ectopic_starts = None

        self.ectopic_beat_thresh = None
        self.ectopic_interval_thresh = None

    def para_config(self, para):
        """
        Configure the RespRateEstimator with the given parameters.

        :param para: Parameters for configuration
        """
        self.ecg_raw = para.ecg_signal_re
        self.ecg_fs = para.fs_system

    def set_config(self, config):
        """
        Set the configuration of the RespRateEstimator.

        :param config: Configuration dictionary
        """
        self.resp_fs = config['resp_fs']
        self.burst_noise_thr = config['burst_noise_thr']
        self.peak_method = config['peak_method']
        self.fusion_method = config['fusion_method']
        self.sd_thresh = config['sd_thresh']
        self.trim_mean_cutoff = config['trim_mean_cutoff']
        self.percentile = config['percentile']
        self.height_prop = config['height_prop']
        self.signals_to_use = config['signals_to_use']
        self.kernel_size = config['kernel_size']
        self.stride_length = config['stride_length']
        self.use_wavelet = config['use_wavelet']

        self.ectopic_beat_thresh = config['ectopic_beat_thresh']
        self.ectopic_interval_thresh = config['ectopic_interval_thresh']

    def train(self, X, y):
        """
        Train the RespRateEstimator. (Placeholder method)

        :param X: Training data
        :param y: Training labels
        """
        pass

    def _check_arrhythmia(self, s, e):
        """
        Check if arrhythmia is above a threshold in the ECG signal interval.

        :param s: Start position of the ECG signal
        :param e: End position (exclusive) of the ECG signal
        :return: True if arrhythmia is above the threshold, False otherwise
        """
        num_beats = 0
        interval_len = 0
        if self.ectopic_beats is not None and len(self.ectopic_beats) > 0:
            num_beats = np.sum((self.ectopic_beats >= s) & (self.ectopic_beats < e))
            if num_beats > self.ectopic_beat_thresh:
                return True

        if self.ectopic_intervals is not None and len(self.ectopic_intervals) > 0:
            i = bisect.bisect_left(self.ectopic_starts, s)
            while i < len(self.ectopic_intervals):
                if self.ectopic_intervals[i][0] >= e:
                    break
                else:
                    interval_len += min(self.ectopic_intervals[i][1], e) - max(self.ectopic_intervals[i][0], s)
                i += 1
        if interval_len > self.ecg_fs * self.ectopic_interval_thresh:
            return True
        return False

    def _ecg_segments(self):
        """
        Generate ECG signal segments based on the defined kernel size and strides.

        :return: Generator yielding ECG segments, end position, and arrhythmia flag
        """
        if self.ecg_raw is None or len(self.ecg_raw) == 0:
            raise StopIteration
        elif len(self.ecg_raw) <= self.kernel_size * self.ecg_fs:
            yield self.ecg_raw, len(self.ecg_raw) - 1, self._check_arrhythmia(0, len(self.ecg_raw))
        else:
            kernel = self.kernel_size * self.ecg_fs
            stride = self.stride_length * self.ecg_fs
            for i in range(kernel, len(self.ecg_raw) + 1, stride):
                s = i - kernel
                yield self.ecg_raw[s:i], i, self._check_arrhythmia(s, i)

    def listen_classifier_out(self, result_group):
        """
        Listen to the classifier output and update the RespRateEstimator.

        :param result_group: Classifier output group
        """
        self.get_classifiers_output(result_group)

    def get_classifiers_output(self, classifier_output):
        """
        Get the output from the arrhythmia classifiers and update the RespRateEstimator.

        :param classifier_output: Output from the classifier group
        :return: self
        """
        self.classifier_output = classifier_output
        beats = []
        intervals = []
        for key in classifier_output:
            arry = classifier_output[key]
            if isinstance(arry, BeatArrhythmia):
                beats += list(arry.episodes_list)
            elif isinstance(arry, IntervalArrhythmia):
                intervals += [tuple(x) for x in arry.onset_offset_list]
        beats = sorted(list(set(beats)))
        intervals.sort()
        merged_intervals = []
        if len(intervals) > 0:
            last_interval = list(intervals[0])
            for i in range(1, len(intervals)):
                if intervals[i][0] <= last_interval[1]:
                    last_interval[1] = max(intervals[i][1], last_interval[1])
                else:
                    merged_intervals.append(last_interval)
                    last_interval = list(intervals[i])
            merged_intervals.append(last_interval)
        self.ectopic_beats = np.array(beats)
        self.ectopic_intervals = intervals
        self.ectopic_starts = [x[0] for x in intervals]
        return self

    def estimate_resp_rate(self):
        """
        Estimate the respiration rate using the generated respiratory signals.

        :return: Respiration rates, individual rates, individual qualities, notes, and positions
        """
        resp = Resp(self.resp_fs, self.burst_noise_thr, signals_to_use=self.signals_to_use)
        rates = []
        all_ind_rates = []
        all_ind_qualities = []
        notes = []
        positions = []
        for ecg_seg, position, is_ectopic in self._ecg_segments():
            resp.generate_resp_signals(ecg_seg, self.ecg_fs, use_wavelet=self.use_wavelet)
            ind_rates, ind_qualities = resp.cal_ind_rates(peak_method=self.peak_method,
                                                          height_prop=self.height_prop,
                                                          trim_mean_cutoff=self.trim_mean_cutoff)
            if self.fusion_method == 'quality':
                rate = quality_fusion(ind_rates, ind_qualities, self.sd_thresh)
            else:
                rate = smart_fusion(ind_rates, self.sd_thresh)

            if is_ectopic:
                note = 'Arrhythmia, accuracy warning'
            else:
                note = None

            positions.append(position)
            rates.append(rate)
            all_ind_rates.append(ind_rates)
            all_ind_qualities.append(ind_qualities)
            notes.append(note)
        return rates, all_ind_rates, all_ind_qualities, notes, positions

    def predict(self, X):
        """
        Predict the respiration rate using the RespRateEstimator.

        :param X: Input data
        :return: RespRateResult object containing the estimated respiration rates, positions, and notes
        """
        resp_rates, ind_rates, qualities, notes, positions = self.estimate_resp_rate()
        return RespRateResult(resp_rates, positions, notes)


class RespRateResult:
    def __init__(self, resp_rates, positions, notes):
        """
        Initialize the RespRateResult object.

        :param resp_rates: Estimated respiration rates
        :param positions: Positions of the respiration rates
        :param notes: Notes associated with the respiration rates
        """
        self.name = 'Respiration_rates'
        self.resp_rates = resp_rates
        self.positions = positions
        self.notes = notes


class Resp:
    def __init__(self, resp_fs, burst_noise_thr, signals_to_use=('bw', 'am', 'fm')):
        """
        Initialize the Resp object.

        :param resp_fs: Respiration sampling frequency
        :param burst_noise_thr: Burst noise threshold
        :param signals_to_use: Signals to use for respiration rate estimation (default: ('bw', 'am', 'fm'))
        """
        self.resp_signals = None
        self.resp_names = None
        self.resp_fs = resp_fs
        self.ecg_time = None
        self.burst_noise_thr = burst_noise_thr
        self.signals_to_use = signals_to_use

    @staticmethod
    def _filter_ecg_signal(ecg_sig, ecg_fs):
        """
        Filter the ECG signal using a lowpass filter at a cutoff frequency of 40 Hz.

        :param ecg_sig: ECG signal
        :param ecg_fs: ECG sampling frequency
        :return: Filtered ECG signal and baseline wander fixed ECG signal
        """
        ecg_sig = detrend(ecg_sig)
        ecg_seg = fix_baseline_wander(ecg_sig, ecg_fs=ecg_fs)
        ecg_sig, _, _ = st.filter_signal(
            signal=ecg_sig,
            ftype="butter",
            band="lowpass",
            order=4,
            frequency=40,
            sampling_rate=ecg_fs)

        return ecg_sig, ecg_seg

    @staticmethod
    def _ectopic_peaks(loc):
        """
        Identify ectopic peaks based on peak/trough locations.

        :param loc: Locations of the peaks/troughs
        :return: Boolean array corresponding to loc, True indicating an ectopic location
        """
        if len(loc) >= 3:
            tk_neg1 = loc[:-2]
            tk = loc[1:-1]
            tk_pos1 = loc[2:]

            r = 2 * np.abs(
                (tk_neg1 - (2 * tk) + tk_pos1)
                / ((tk_neg1 - tk) * (tk_neg1 - tk_pos1) * (tk - tk_pos1)))
            thresh = min([4.3 * np.std(r[r < np.inf]), 0.5])
            temp = np.concatenate([[0], r, [0]])
        else:
            temp = np.ones_like(loc)
            thresh = 0
        temp = temp > thresh
        return temp

    def _add_signal(self, sig, name):
        """
        Add a respiratory signal to the Resp object.

        :param sig: Respiratory signal
        :param name: Name of the respiratory signal
        """
        self.resp_signals.append(sig)
        self.resp_names.append(name)

    def generate_resp_signals(self, ecg_sig, ecg_fs, use_wavelet=True):
        """
        Generate respiratory signals based on the specified signals to use.

        :param ecg_sig: ECG signal
        :param ecg_fs: ECG sampling frequency
        :param use_wavelet: Whether to use DWT-based filtering (default: True)
        :return: self
        """
        ecg_sig, ecg_seg = self._filter_ecg_signal(ecg_sig, ecg_fs)

        self.resp_signals = []
        self.resp_names = []

        r_loc, q_loc = get_qr_peaks(ecg_sig, ecg_fs)

        if not (r_loc is None or len(r_loc) == 0):
            temp1 = self._ectopic_peaks(r_loc)
            temp2 = self._ectopic_peaks(q_loc)

            r_loc[temp1 | temp2] = -1 * (np.max([q_loc, r_loc]) + 1000)
            q_loc[temp1 | temp2] = -1 * (np.max([q_loc, r_loc]) + 1000)

            if 'bw' in self.signals_to_use:
                bw = RespSignal.bw(ecg_sig, r_loc=r_loc, q_loc=q_loc,
                                   sampling_rate=ecg_fs, resampling_rate=self.resp_fs)
                self._add_signal(bw, 'bw')

            if 'am' in self.signals_to_use:
                am = RespSignal.am(ecg_sig, r_loc=r_loc, q_loc=q_loc,
                                   sampling_rate=ecg_fs, resampling_rate=self.resp_fs)
                self._add_signal(am, 'am')

            if 'fm' in self.signals_to_use:
                fm = RespSignal.fm(r_loc=r_loc, sampling_rate=ecg_fs, resampling_rate=self.resp_fs)
                self._add_signal(fm, 'fm')

            if 'berger_fm' in self.signals_to_use:
                fm = RespSignal.berger_fm(r_loc=r_loc, sampling_rate=ecg_fs, resampling_rate=self.resp_fs)
                self._add_signal(fm, 'berger_fm')

            if 'freq' in self.signals_to_use:
                freq = RespSignal.freq(signal=ecg_sig, sampling_rate=ecg_fs, resampling_rate=self.resp_fs)
                self._add_signal(freq, 'freq')

            for sig in self.resp_signals:
                if sig is not None:
                    sig.filter(use_wavelet)

        return self

    def cal_ind_rates(self, peak_method, height_prop=None, trim_mean_cutoff=0.):
        """
        Calculate individual respiration rates and qualities for each respiratory signal.
    
        :param peak_method: Peak detection method
        :param height_prop: Height proportion for peak detection (default: None)
        :param trim_mean_cutoff: Trim mean cutoff for rate calculation (default: 0.0)
        :return: Individual respiration rates and qualities
        """
        ind_rates = np.zeros(len(self.signals_to_use))
        ind_qualities = np.zeros(len(self.signals_to_use))
        
        for i, resp_sig in enumerate(self.resp_signals):
            if resp_sig is not None:
                resp_sig.cal_resp_rate(peak_method,
                                       height_prop=height_prop,
                                       trim_mean_cutoff=trim_mean_cutoff)
    
                ind_rates[i] = resp_sig.resp_rate
                ind_qualities[i] = resp_sig.quality
            else:
                ind_rates[i] = 0
                ind_qualities[i] = 0
        
        return ind_rates, ind_qualities

class RespSignal:
    def __init__(self, signal, fs, timestamp=None):
        """
        Initialize the RespSignal object.
    
        :param signal: Respiratory signal
        :param fs: Sampling frequency
        :param timestamp: Timestamp of the respiratory signal (default: None)
        """
        self.signal = signal
        self.fs = fs
    
        self.timestamp = timestamp
        self.filtered = None
    
        self.resp_rate = None
        self.quality = None
    
    def resample(self, target_fs):
        """
        Resample the respiratory signal to the target sampling frequency.
    
        :param target_fs: Target sampling frequency
        :return: self
        """
        self.signal = resampy.resample(self.signal, self.fs, target_fs)
        self.fs = target_fs
        return self
    
    def filter(self, use_wavelet=True):
        """
        Filter the respiratory signal.
    
        :param use_wavelet: Whether to use DWT-based filtering (default: True)
        :return: self
        """
        self.filtered = filter_resp_signal(self.signal, self.fs, use_wavelet=use_wavelet)
        return self
    
    def cal_resp_rate(self, peak_method, height_prop=None, trim_mean_cutoff=0.):
        """
        Calculate the respiration rate and quality of the respiratory signal.
    
        :param peak_method: Peak detection method
        :param height_prop: Height proportion for peak detection (default: None)
        :param trim_mean_cutoff: Trim mean cutoff for rate calculation (default: 0.0)
        :return: self
        """
        if self.filtered is None:
            self.resp_rate = 0
            self.quality = 0
            return self
        if peak_method == 'chan':
            peak_results = chan_peak(self.filtered, self.fs, height_prop=height_prop)
            quality_result = chan_quality(self.filtered,
                                          peaks=peak_results['peaks'],
                                          troughs=peak_results['troughs'],
                                          num_maxima=peak_results['num_max'],
                                          num_minima=peak_results['num_min'])
        elif peak_method == 'count_orig':
            peak_results = count_orig(self.filtered, percent=75, height_prop=height_prop)
            quality_result = interval_quality(self.filtered,
                                              intervals=peak_results['intervals'],
                                              peaks=peak_results['peaks'])
        elif peak_method == 'zero_crossing':
            peak_results = zero_crossing(self.filtered)
            quality_result = interval_quality(self.filtered,
                                              intervals=peak_results['intervals'],
                                              peaks=peak_results['peaks'])
        else:
            raise Exception("undefined peak finding method")
        self.peak_result = peak_results
        self.quality_result = quality_result
        if len(peak_results['intervals']) == 0:
            self.resp_rate = 0
            self.quality = 0
        else:
            self.resp_rate = 60 * self.fs / trim_mean(peak_results['intervals'], proportiontocut=trim_mean_cutoff)
            self.quality = quality_result['quality']
        return self
    
    @classmethod
    def bw(cls, signal, r_loc, q_loc, sampling_rate, resampling_rate):
        """
        Generate a respiratory signal from the baseline wandering of the ECG signal.
    
        :param signal: Filtered ECG signal
        :param r_loc: Locations of R-peaks
        :param q_loc: Locations of Q-peaks or troughs
        :param sampling_rate: Sampling rate of the ECG signal
        :param resampling_rate: Resampling rate of the respiratory signal
        :return: RespSignal object representing the baseline wandering respiratory signal
        """
        r_loc = r_loc[r_loc >= 0]
        q_loc = q_loc[q_loc >= 0]
        timestamp = 0.5 * (r_loc + q_loc) / sampling_rate
        amp = 0.5 * (signal[r_loc] + signal[q_loc])
        amp /= np.mean(signal[r_loc] - signal[q_loc])
        if len(amp) < 2:
            return None
        resampled_time, resampled_sig = tools.resampling(timestamp, amp, resampling_rate)
    
        bw_sig = cls(resampled_sig, resampling_rate, resampled_time)
        return bw_sig
    
    @classmethod
    def am(cls, signal, r_loc, q_loc, sampling_rate, resampling_rate):
        """
        Generate a respiratory signal from the amplitude modulation of the ECG signal.
    
        :param signal: Filtered ECG signal
        :param r_loc: Locations of R-peaks
        :param q_loc: Locations of Q-peaks or troughs
        :param sampling_rate: Sampling rate of the ECG signal
        :param resampling_rate: Resampling rate of the respiratory signal
        :return: RespSignal object representing the amplitude modulation respiratory signal
        """
        r_loc = r_loc[r_loc >= 0]
        q_loc = q_loc[q_loc >= 0]
        timestamp = 0.5 * (r_loc + q_loc) / sampling_rate
        amp = signal[r_loc] - signal[q_loc]
        amp /= np.mean(amp)
        if len(amp) < 2:
            return None
        resampled_time, resampled_sig = tools.resampling(timestamp, amp, resampling_rate)
    
        am_sig = cls(resampled_sig, resampling_rate, resampled_time)
        return am_sig
    
    @classmethod
    def fm(cls, r_loc, sampling_rate, resampling_rate):
        """
        Generate a respiratory signal from the frequency modulation of the ECG signal by respiratory activities.
    
        :param r_loc: Locations of R-peaks
        :param sampling_rate: Sampling rate of the ECG signal
        :param resampling_rate: Resampling rate of the respiratory signal
        :return: RespSignal object representing the frequency modulation respiratory signal
        """
        timestamp = 0.5 * (r_loc[1:] + r_loc[:-1]) / sampling_rate
        intervals = (r_loc[1:] - r_loc[:-1]) / sampling_rate
        timestamp = timestamp[(r_loc[1:] >= 0) & (r_loc[:-1] >= 0)]
        intervals = intervals[(r_loc[1:] >= 0) & (r_loc[:-1] >= 0)]
        intervals /= np.mean(intervals)
        if len(timestamp) < 2:
            return None
        resampled_time, resampled_sig = tools.resampling(
            timestamp, intervals, resampling_rate
        )
    
        fm_sig = cls(resampled_sig, resampling_rate, resampled_time)
        return fm_sig
    
    @classmethod
    def berger_fm(cls, r_loc, sampling_rate, resampling_rate):
        """
        Generate a respiratory signal from the frequency modulation of the ECG signal using Berger's algorithm.
    
        :param r_loc: Locations of R-peaks
        :param sampling_rate: Sampling rate of the ECG signal
        :param resampling_rate: Resampling rate of the respiratory signal
        :return: RespSignal object representing the frequency modulation respiratory signal (Berger's method)
        """
        intervals = np.diff(r_loc) / sampling_rate
        intervals /= np.mean(intervals)
        window = sampling_rate / resampling_rate
        window_values = []
        r_idx = 0
        for w_start in np.arange(r_loc[0], r_loc[-1] - window, window):
            w_end = w_start + window
            curr_value = 0
            temp = w_start
            while r_loc[r_idx + 1] < w_end:
                curr_value += (r_loc[r_idx + 1] - temp) / intervals[r_idx]
                temp = r_loc[r_idx + 1]
                r_idx += 1
            curr_value += (w_end - temp) / intervals[r_idx]
            window_values.append(curr_value)
        window_values = np.array(window_values)
        resampled_sig = window_values[:-1] + window_values[1:]
        resampled_time = r_loc[0] / sampling_rate + np.arange(1, len(resampled_sig) + 1) / resampling_rate
    
        fm_sig = cls(resampled_sig, resampling_rate, resampled_time)
        return fm_sig
    
    @classmethod
    def freq(cls, signal, sampling_rate, resampling_rate):
        """
        Generate a respiratory signal by filtering the ECG signal to the frequency bands of respiratory signals.
    
        :param signal: Filtered ECG signal
        :param sampling_rate: Sampling rate of the ECG signal
        :param resampling_rate: Resampling rate of the respiratory signal
        :return: RespSignal object representing the respiratory signal obtained by filtering
        """
        freq_sig = cls(signal, sampling_rate)
        freq_sig.resample(resampling_rate)
        freq_sig.signal /= np.mean(freq_sig.signal)
    
        return freq_sig
