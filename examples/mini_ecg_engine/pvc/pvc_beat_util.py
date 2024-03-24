#!/usr/bin/env python
#-*- coding:utf8 -*-

import yaml
import json
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster
from operator import itemgetter
import itertools
from scipy.signal import lfilter
import peakutils
import math

# Start of merged code from utils.py
def ecg_bandpass_filter(signal, lowpass, highpass, fs):
    '''
    This function works like a lowpass, highpass and bandpass filter on input signal

    :param signal: ndarray, input ECG signal
    :param lowpass: the frequency of low frequency sigal to be cut
    :param highpass: the frequency of high frequency signal to be cut
    :param fs: sampling frequency

    :returns: filtered_signal - ndarray, output filtered signal
    '''

    if lowpass != 0:
        cb, ca = butter(2, lowpass/(fs/2.0))
        lowpass_signal = filtfilt(cb, ca, signal)
        
        if highpass == 0: # lowpass filter
            return lowpass_signal
        else: # bandpass filter
            # assert highpass < lowpass
            cb, ca = butter(2, highpass/(fs/2.0))
            baseline_signal = filtfilt(cb, ca, signal)
            return lowpass_signal - baseline_signal
    else: 
        if highpass == 0: # do nothing
            return signal
        else: # highpass filter
            cb, ca = butter(2, highpass/(fs/2.0))
            baseline_signal = filtfilt(cb, ca, signal)
            return signal - baseline_signal

def normalize2one(a_vector):
    '''
    Normalize the a_vector to one

    :param a_vector: a vector
    :type a_vector: ndarray
    :returns: normalized vector
    :rtype: ndarray
    '''
    return a_vector / max(abs(a_vector))


def get_up_down_slope(poss_signal):
    '''
    Find the up slope and down slope point based on the input poss_signal

    :param poss_signal: vector of 0 and 1, 1 means above threshold, 0 means below threshold
    :returns:
        - **up_slope**: a vector of index of 0-1 turnning point
        - **down_slope**: a vector of index of 1-0 turning point
    :rtype: vector, vector

    .. note:: size of **up_slope** and **down_slope** will always match due to the left/right
              padding of 0
    '''

    up_slope = np.nonzero(np.diff(np.concatenate(([0], poss_signal))) == 1)[0]
    down_slope = np.nonzero(np.diff(np.concatenate((poss_signal, [0]))) == -1)[0]
    return up_slope, down_slope

class Peak(object):
    '''
    This class represents the peaks detected in ECG signal, like P, Q, R, S, T wave

    :param peak_loc: location of the peak
    :type peak_loc: ndarray
    :param peak_value: value of the peak
    :type peak_value: ndarray
    :param label: (option) label of the peak such as 'P', 'Q', 'R', 'S', 'T'
    :type label: str
    :param fs: sampling rate
    :type fs: int
    '''

    def __init__(self, peak_loc, peak_value, label=None, fs=0):
        assert peak_loc.size == peak_value.size
        self.loc = peak_loc
        self.value = peak_value
        self.label = label
        self.fs = fs

    @property
    def size(self):
        return self.loc.size

    @property
    def ts(self):
        '''Return the timestamp of the peak location in millisecond'''
        return self.loc / self.fs * 1000

    @property
    def interval(self):
        '''Return a sequence of peak interval in sample'''
        return np.diff(self.loc)

    @property
    def interval_ts(self):
        '''Return a sequence of peak interval in time (ms)'''
        return self.interval / self.fs * 1000

    @property
    def avg_interval(self):
        return np.mean(self.interval)

    @property
    def avg_interval_ts(self):
        return np.mean(self.interval_ts)

    def irregular_interval(self):
        '''Get irregular interval from the interval sequence

        irregular RR beats is defined as :
        any single beat with RR-interval +/- 10% from prior beat

        :returns:
            - num_irr: number of irregular interval
            - percent_irr: percentage of irregular interval
            - irr_loc: location of irregular interval
        '''
        #mark_r_int = abs(np.diff(self.interval)) / self.interval[1:] > 0.1
        mark_r_int = np.zeros(len(self.loc), dtype='int')
        for i in range(1,len(self.loc)-1):
            if (self.interval[i] - self.interval[i-1]) / self.interval[i] > 0.1:
                mark_r_int[i] = 1

        num_irr = sum(mark_r_int)
        percent_irr = num_irr / self.interval.size * 100
        # first 2 are ignored due to double diff
        return num_irr, percent_irr, self.loc[2:][mark_r_int]

    @property
    def poincare(self):
        '''
        Return Poincare data for plot in terms of xvalue, yvalue
        '''
        return self.interval_ts[:-1], self.interval_ts[1:]

    def num_of_samples(self, start, size=0, end=0):
        '''
        Return number of samples corresponding to `size` of beats from `start` beat

        .. note:: size = end - start
        return number of samples for beat 3 and beat 4
        '''
        if end >= start:
            size = end - start

        return self.loc[start + size] - self.loc[start]

    def __repr__(self):
        return '{name}-Peak(SIZE={size}, FS={fs})'.format(name=self.label, size=self.size, fs=self.fs)

    def __str__(self):
        return '{name}-Peak(SIZE={size}, FS={fs})'.format(name=self.label, size=self.size, fs=self.fs)

class PeakMetrics(object):
    '''
    This class represents all types of peaks detected in ECG signal, like P, Q, R, S, T

    :param p_peak: p peak
    :type p_peak: Peak
    :param q_peak: q peak
    :type q_peak: Peak
    :param r_peak: r peak
    :type r_peak: Peak
    :param s_peak: s peak
    :type s_peak: Peak
    :param t_peak: t peak
    :type t_peak: Peak
    :param fs: (option) sampling rate
    :type fs: int
    :param tom: (option) ptompkins filter 6 signals output
    :type tom: tuple
    '''

    def __init__(self, p_peak=None, q_peak=None, r_peak=None, s_peak=None, t_peak=None, fs=0, tom=None,
                 count_p = None, r_label_exc=None, ex_info=None):
        self.p = p_peak
        self.q = q_peak
        self.r = r_peak
        self.s = s_peak
        self.t = t_peak

        # self.count_p_inv = ex_mtx["count_p_inv"] if ex_mtx else None
        self.r_label_exc = r_label_exc if r_label_exc else np.empty(0).astype(int)# This is the set of the r labels excluded after the qrs result. 

        self.count_p = count_p

        self.fs = r_peak.fs if r_peak is not None else fs
        self.tom = tom

        self.exinfo = ex_info if ex_info else dict()

    @property
    def count_p_inv(self):
        return 1 - self.count_p[:-1] if self.count_p is not None else None

    @property
    def peak_size(self):
        return self.r.size if self.r is not None else 0

    @property
    def signal_size(self):
        return self.tom[0].size if self.tom is not None else 0

    def get_segment_by_num_of_beats(self, start_beat, num_beats):
        '''
        Get a segment of original signal by number of beats. The segment starts
        from start_beat. The segment contains `num_beats` beats.

        .. note:: The last beat's peak is included

        :param start_beat: R peak loc, start from 0
        :param num_beats: number of beats (r peaks)
        :returns: ecg_seg
        :rtype: ndarray
        '''
        return self.tom[0][self.r.loc[start_beat]:self.r.loc[start_beat + num_beats] + 1]

    @property
    def qrs(self):
        '''Get QRS interval sequence'''
        return self.s.loc - self.q.loc

    @property
    def qrs_ts(self):
        '''Get QRS interval sequence in ms'''
        return self.qrs / self.fs * 1000

    @property
    def qt(self):
        '''Get QT interval sequence

        .. note::

        Q and T may not align
        '''
        _qt = self.t.loc - self.q.loc
        return _qt[:-1] if self.t.loc[-1] == 0 else _qt

    @property
    def qt_ts(self):
        '''Get QT interval sequence in ms'''
        return self.qt / self.fs * 1000

    @property
    def pr(self):
        '''Get PR interval sequence

        .. note::
        P and R may not align
        '''
        assert self.count_p is not None
        assert self.count_p.size == self.p.size == self.r.size

        _pr = np.zeros(self.count_p.size)
        for i in range(self.count_p.size):
            if self.count_p[i] == 1:
                for r_loc in self.r.loc:
                    if r_loc > self.p.loc[i]:
                        _pr[i] = r_loc - self.p.loc[i]
                        break

        return _pr

    @property
    def pr_ts(self):
        '''Get PR interval sequence in ms'''
        return self.pr / self.fs * 1000

    @property
    def st(self):
        '''Get ST interval sequence'''
        # matlab: if the first instance of T_loc is less then S_loc
        # discard the 1st locations
        _st = self.t.loc - self.s.loc
        return _st[:-1] if self.t.loc[-1] == 0 else _st

    @property
    def st_ts(self):
        '''Get ST interval sequence in ms'''
        return self.st / self.fs * 1000

    @property
    def qt_c_instants(self):
        '''Get qt_c_instants in ms
        QTc = QT / sqrt(RR)
        refer to this link on how to calculate QTc (corrected QT interval)
        https://lifeinthefastlane.com/eponymictionary/bazett-formula/
        '''
        return self.qt_ts[:self.r.size - 1] / np.sqrt(self.r.interval_ts / 1000)

    def get_avg_rr(self):
        '''Calculate average RR interval in ms'''
        return self.r.avg_interval_ts

    def get_avg_hr(self):
        '''Get average heart rate in min'''
        return 60 * 1000 / self.get_avg_rr()

    def get_avg_pr(self):
        '''Calculate average PR interval in ms (interval between all P wave - R wave)
        '''
        return np.mean(self.pr_ts)

    def get_avg_qt(self):
        '''Calculate average QT interval'''
        return np.mean(self.qt_ts)

    def get_avg_qtc(self):
        '''Calculate average QTc interval'''
        return self.get_avg_qt() / np.sqrt(self.get_avg_rr() / 1000)

    def get_avg_qrs(self):
        '''Calculate average QRS interval'''
        return np.mean(self.qrs_ts)

    def get_avg_st(self):
        '''Calculate average ST interval'''
        return np.mean(self.st_ts)

    @property
    def hrv_stats(self):
        '''
        calculate HRV stats

        :param hrv_data: HRV data
        :returns:
            - **mrr**:
            - **sdnn**:
            - **rmssd**:
        '''
        hrv_data = self.r.interval_ts

        ### exclude beats if peak_metrics.r_label_exc is not empty
        # set hrv_data to NaN if there is r peak excluded in between of two r peak
        if self.r_label_exc is not None and self.r_label_exc.size > 0:
            try:
                hrv_data[np.searchsorted(self.r.loc, self.r_label_exc) - 1] = np.nan
            except IndexError:
                logger.warning('hrv_data out of index due to r_label_exc to be last of r loc')

        mrr, sdnn = np.nanmean(hrv_data[1:]), np.nanstd(hrv_data[1:])
        n_rr_len = hrv_data.size - 1

        hrv_data = hrv_data[np.isfinite(hrv_data)]
        succ_diffs = np.diff(hrv_data[1:])
        rmssd = np.sqrt(sum(succ_diffs ** 2)/(n_rr_len - 2))

        return  HRVStats(mrr, sdnn, rmssd)

    @property
    def hrv_data_fun(self):
        '''
        calculate HRV stats

        :param hrv_data: HRV data
        :returns:
            - **mrr**:
            - **sdnn**:
            - **rmssd**:
        '''
        hrv_data = self.r.interval_ts
        if self.r_label_exc is not None and self.r_label_exc.size > 0:
            try:
                hrv_data[np.searchsorted(self.r.loc, self.r_label_exc) - 1] = np.nan
            except IndexError:
                logger.warning('hrv_data out of index due to r_label_exc to be last of r loc')

        mrr, sdnn = np.nanmean(hrv_data[1:]), np.nanstd(hrv_data[1:n_rr_len = hrv_data.size - 1
        hrv_data = hrv_data[np.isfinite(hrv_data)]
        return hrv_data

    @property
    def hrv_datatime(self):
        hrv_datatime = self.r.loc[2:] / self.fs
        return hrv_datatime

    def pac_detector(self):
        pass

    def pause_detector(self):
        pass

    def svt_detector(self):
        pass

    def pvc_detector(self):
        pass

    def find_indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def vf_detector(self):
        pass

    def af_computation(self):
        pass

    def __repr__(self):
        return '{name}(P({p}), Q({q}), R({r}), S({s}), T({t}), FS({fs}))'.format(
            name=self.__class__.__name__, p=self.p.size if self.p else 0,
            q=self.q.size if self.q else 0, r=self.r.size if self.r else 0,
            s=self.r.size if self.s else 0, t=self.t.size if self.t else 0,
            fs=self.fs)

    def __str__(self):
        return '{name}(P({p}), Q({q}), R({r}), S({s}), T({t}), FS({fs}))'.format(
            name=self.__class__.__name__, p=self.p.size if self.p else 0,
            q=self.q.size if self.q else 0, r=self.r.size if self.r else 0,
            s=self.r.size if self.s else 0, t=self.t.size if self.t else 0,
            fs=self.fs)

class PanTompkins(object):
    """
    This class implements the algorithm found in the following paper:

    PAN.J, TOMPKINS. W.J,"A Real-Time QRS Detection Algorithm" IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. BME-32, NO. 3, MARCH 1985

    The current algorithm is composed of the following steps:

    - filter phase.
    - qrs detection
    - t wave detection
    - p wave detection
    """

    def __init__(self, sig, fs, conf=None):
        self.sig = sig
        self.fs = fs
        self.conf = conf
        self.peak_metrics = PeakMetrics(fs=self.fs)
        self.inv_r_peak_locator_flag = conf.get("inv_r_peak_flag",False) if conf is not None else False
        self.inv_r_peak_threshold = conf.get("inv_r_peak_locator_threshold",2.5) if conf is not None else 2.5
        self.inv_r_peak_locator_window_size = conf.get("inv_r_peak_locator_window_size",1) if conf is not None else 1
        
        ### As of May 2022, the filters for the Pan Tompkins Algorithm used for peak detection are modified to reflect the needs of the clinical care team.
        ### For legacy testing purposes, we have left the original set of filters in the code. But for future use, the Pan Tompkins algorithm expects the following configuation key:
        ### "pan_tompkins_filter" to be set to "modified" for the new changes to be applied.
        ### Otherwise, the unmodified legacy Pan Tompkins algorithm will be used instead.

        self.pan_tompkins_filter = conf.get("pan_tompkins_filter","legacy") if conf is not None else "legacy"
        self._filter_ptompkins = self.choose_filter(self.pan_tompkins_filter)
        if self.inv_r_peak_threshold is None or self.inv_r_peak_locator_window_size is None:
            self.inv_r_peak_locator_window_size = 1  

    def choose_filter(self,filter_option=None):
        if isinstance(filter_option,str):
            if filter_option.lower() == "legacy":
                return self._filter_ptompkins_legacy
            if filter_option.lower() == "modified":
                return self._filter_ptompkins_modified
            if filter_option.lower() not in ["legacy","modified"]:
                raise Exception('Filter choice must be one of ["legacy","modified",None]')
        elif filter_option is None:
            return self._filter_ptompkins_legacy 
        else:
            raise Exception('Filter choice must be one of ["legacy","modified",None]')

    def detect(self,components_to_detect='pqrst'):
        """
        Detects the components of the wave desired.
         - qrs is detected, then
         - t wave is detected, then
         - p wave is detected.

        Therefore, `qrs.detect(components_to_detect='t')` results in a peak_metrics that also contains qrs information.

        :param components_to_detect:  String
        :return:
        """

        self._filter_ptompkins()

        if "q" in components_to_detect or 'r' in components_to_detect or 's' in components_to_detect:
            self.detect_qrs()
        if 't' in components_to_detect:
            self.detect_t_wave()
        if 'p' in components_to_detect:
            self.detect_p_wave()

    def _filter_ptompkins_legacy(self):  # Filter_PTompkins
        '''
        REMARK: OLD Implementation of Pan Tompkins filter!!

        Refer to this paper: http://ieeexplore.ieee.org/abstract/document/4122029/

        Main steps:

            1. Cancel DC drift
            2. Apply LP + HP
            3. Derivative
            4. Squaring
            5. Moving Window

        LPF transfer function: `(1-z^-6)^2/(1-z^-1)^2`

        HPF transfer function: `HPF = Allpass-(Lowpass) = z^-16-[(1-z^-32)/(1-z^-1)]`

        ..note::
            - x2_normalized is delayed 6 samples from x1_normalized
            - x3_normalized is delayed 16 samples from x2_normalized
            - x4_normalized and x5_normalized and x6_normalized align with x3_normalized

        :param selfsig: input ecg signal
        :param sample_rate: sampling rate of the input signal
        :returns:
            - **x1_normalized** original signal
            - **x2_normalized** low pass filtered signal from x1_normalized
            - **x3_normalized** high pass filtered signal from x2_normalized
            - **x4_normalized** derivative filtered singal from x3_normalized
            - **x5_normalized** squaring signal from x4_normalized
            - **x6_normalized** moving window integrated signal from x5_normalized
        :rtype: ndarray
        '''
        # step 1: cancel DC drift then normalize
        # ++ chenhao: revise later
        self.x1 = normalize2one(self.sig - np.mean(self.sig))
        # print(np.size(ecg_signal))
        # step 2: low/high pass filter
        # low pass filtering, len of b: 13, delay 6, actually is 5
        b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]  # TODO reveiw for different fs
        a = [1, -2, 1]  # TODO reveiw for different fs

        # tranfer function of LPF
        h_lp = lfilter(b, a, np.concatenate(([1], np.zeros(12))))
        self.x2 = normalize2one(np.self.x2 = normalize2one(np.convolve(self.x1, h_lp))

        # high pass filtering, len of b: 33, delay 16
        b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32,
             -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # TODO reveiw for different fs
        a = [1, -1]  # TODO reveiw for different fs

        h_hp = lfilter(b, a, np.concatenate(([1], np.zeros(32))))
        self.x3 = normalize2one(np.convolve(self.x2, h_hp))

        # step 3: derivative filter
        # make impulse response, delay 2
        h = np.array([-1, -2, 0, 2, 1]) / 8  # TODO reveiw for different fs
        delay = 2
        self.x4 = normalize2one(np.convolve(self.x3, h)[delay:delay + self.sig.size])

        # step 4: squaring
        self.x5 = normalize2one(self.x4 ** 2)

        # step 5: Moving window
        # make impulse response, delay 15
        # NOTE: if sampling rate is 200Hz, the moving window has to be 30 samples wide
        # based on Pan-tompkins algorithm
        # NOTE: this 30 samples is determined empirically (by experience rather than theory)
        # try to resample to 200Hz when you run this algo
        h = np.ones(31) / 31
        delay = 15
        self.x6 = normalize2one(np.convolve(self.x5, h)[delay:delay + self.sig.size])
        tom = (self.x1, self.x2, self.x3, self.x4, self.x5, self.x6)
        self.peak_metrics.tom = tom


    def _filter_ptompkins_modified(self):  # Filter_PTompkins
        '''
        Implementation of Pan Tompkins filter

        Refer to this paper: http://ieeexplore.ieee.org/abstract/document/4122029/

        Main steps:

            1. Cancel DC drift
            2. Apply LP + HP
            3. Derivative
            4. Squaring
            5. Moving Window

        LPF transfer function: `(1-z^-6)^2/(1-z^-1)^2`

        HPF transfer function: `HPF = Allpass-(Lowpass) = z^-16-[(1-z^-32)/(1-z^-1)]`

        ..note::
            - x2_normalized is delayed 6 samples from x1_normalized
            - x3_normalized is delayed 16 samples from x2_normalized
            - x4_normalized and x5_normalized and x6_normalized align with x3_normalized

        :param selfsig: input ecg signal
        :param sample_rate: sampling rate of the input signal
        :returns:
            - **x1_normalized** original signal
            - **x2_normalized** low pass filtered signal from x1_normalized
            - **x3_normalized** high pass filtered signal from x2_normalized
            - **x4_normalized** derivative filtered singal from x3_normalized
            - **x5_normalized** squaring signal from x4_normalized
            - **x6_normalized** moving window integrated signal from x5_normalized
        :rtype: ndarray
        '''
        # step 1: cancel DC drift then normalize
        # ++ chenhao: revise later
        self.x1 = normalize2one(self.sig - np.mean(self.sig))
        # print(np.size(ecg_signal))
        # step 2: low/high pass filter
        # low pass filtering, len of b: 13, delay 6, actually is 5
        b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]  # TODO reveiw for different fs
        a = [1, -2, 1]  # TODO reveiw for different fs

        # tranfer function of LPF
        h_lp = lfilter(b, a, np.concatenate(([1], np.zeros(12))))
        self.x2 = normalize2one(np.convolve(self.x1, h_lp))

        # high pass filtering, len of b: 33, delay 16
        b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        a = [1, 1]

        h_hp = lfilter(b, a, np.concatenate(([1], np.zeros(32))))
        self.x3 = normalize2one(np.convolve(self.x2, h_hp))

        # step 3: derivative filter
        # make impulse response, delay 2
        b = np.array([1, 2, 0, -2, -1]) / 8
        a = [1]
        delay = 2
        slope_fil = lfilter(b, a, np.concatenate(([1], np.zeros(4))))
        self.x4 = normalize2one(np.convolve(self.x3, slope_fil)[delay:delay + self.sig.size])

        # step 4: squaring
        self.x5 = normalize2one(self.x4 ** 2)

        # step 5: Moving window
        # make impulse response, delay 15
        # NOTE: if sampling rate is 200Hz, the moving window has to be 30 samples wide
        # based on Pan-tompkins algorithm
        # NOTE: this 30 samples is determined empirically (by experience rather than theory)
        # try to resample to 200Hz when you run this algo
        h = np.ones(31) / 31
        delay = 15
        self.x6 = normalize2one(np.convolve(self.x5, h)[delay:delay + self.sig.size])
        tom = (self.x1, self.x2, self.x3, self.x4, self.x5, self.x6)
        self.peak_metrics.tom = tom

    def detect_p_wave(self):
        """
         Detects the p wave.

        Example:
        >>> qrs = PanTompkins(sig,fs)
        >>> qrs.detect_p_wave()
        >>> print(qrs.peak_metrics.p)

        (qrs and t waves are also detected and stored on peak_metrics)

        :return:
        """
        # find P wave based on QRS T
        if not self.peak_metrics.t:
            self.detect_t_wave()
        p_peak1, count_p, p_loc_left = self._ppeak(self.x1, self.x2, self.peak_metrics.q, self.peak_metrics.r,
                                                   self.peak_metrics.s, self.peak_metrics.t,
                                                   self.t_inv,
                                                   self.t_inv_rt_loc,
                                                   self.t_loc_right,
                                                   self.left, self.right, self.fs)

        p_peak1.fs = self.fs
        self.peak_metrics.p=p_peak1
        self.peak_metrics.count_p = count_p

    def detect_t_wave(self):
        """
        Detects the t wave.

        Example:
        >>> qrs = PanTompkins(sig,fs)
        >>> qrs.detect_t_wave() #Detect t waves
        >>> print(qrs.peak_metrics.t)

        (qrs wave is also detected and stored on peak_metrics)
        """

        # find T wave based on QRS
        if (not self.peak_metrics.q) or (not self.peak_metrics.r) or (not self.peak_metrics.s):
            self.detect_qrs()

        t_peak1, width_t, t_inv, t_inv_rt_loc, t_loc_right = self._tpeak(self.x1, self.x2, self.peak_metrics.q,
                                                                         self.peak_metrics.r, self.peak_metrics.s,
                                                                         self.left,
                                                                         self.right,
                                                                         self.fs)
        t_peak1.fs = self.fs

        self.peak_metrics.t=t_peak1

        self.t_inv = t_inv
        self.t_inv_rt_loc = t_inv_rt_loc
        self.t_loc_right = t_loc_right

        return t_inv, t_inv_rt_loc, t_loc_right, t_peak1, width_t

    def detect_qrs(self):
        """
        Detects the qrs complex.

        Example:
        >>> qrs = PanTompkins(sig,fs)
        >>> qrs.detect_qrs()
        >>> print(qrs.peak_metrics.q)
        >>> print(qrs.peak_metrics.r)
        >>> print(qrs.peak_metrics.s)
        """
        if not hasattr(self.peak_metrics,'tom'):
            self._filter_ptompkins()

        threshold = ((np.mean(self.x6) + np.median(self.x6)) / 2) * max(self.x6)
        poss_reg = (self.x6 > threshold).astype(int)
        # left:  left side (up slope) of the wave crossing the threshold
        # right: right side (down slope) of the wave crossing the threshold
        left, right = get_up_down_slope(poss_reg)
        for m in range(left.size):

            seg_x6 = self.x6[left[m]:right[m] + 1]  # diff with matlab, the second index is exclusive
            seg_x6_max = max(seg_x6)
            seg_threshold = 0.5 * seg_x6_max if seg_x6_max < 0.55 else 0.35 * seg_x6_max

            seg_poss = (seg_x6 > seg_threshold).astype(int)

            additional_left, additional_right = get_up_down_slope(seg_poss)

            if additional_left.size == additional_right.size == 1:
                # note: here is a value update in place, cannot break the line
                left[m], right[m] = left[m] + additional_left, left[m] + additional_right
        # cancel delay because of Low Pass and High Pass
        if self.pan_tompkins_filter == "modified":
            left -= (6 + 16 + 2)
            right -= (6 + 16 + 2)
        else:
            left -= (6 + 16)
            right -= (6 + 16)       
        # find QRS wave peaks
        q_peak, r_peak, s_peak = self._qrspeaks(self.x1, left, right)
        q_peak1, r_peak1, s_peak1, left, right = self._correct_peak(q_peak, r_peak, s_peak, left, right, self.x6,
                                                                    self.fs)

        q_peak1.fs = r_peak1.fs = s_peak1.fs = self.fs

        self.peak_metrics.q = q_peak1
        self.peak_metrics.r = r_peak1
        self.peak_metrics.s = s_peak1
        self.left = left # this might be used in some test
        self.right = right

    def _get_up_down_slope(self, poss_signal):
        '''
        Find the up slope and down slope point based on the input poss_signal

        :param poss_signal: vector of 0 and 1, 1 means above threshold, 0 means below threshold
        :returns:
            - **up_slope**: a vector of index of 0-1 turnning point
            - **down_slope**: a vector of index of 1-0 turning point
        :rtype: vector, vector

        .. note:: size of **up_slope** and **down_slope** will always match due to the left/right
                  padding of 0
        '''

        up_slope = np.nonzero(np.diff(np.concatenate(([0], poss_signal))) == 1)[0]
        down_slope = np.nonzero(np.diff(np.concatenate((poss_signal, [0]))) == -1)[0]
        return up_slope, down_slope

    def _qrspeaks(self,x1, left, right):
        '''
        Detect QRS peak

        :param x1: original ecg signal, first output of tompkins filter
        :param left: vector of refined index of up-slope point
        :param right: vector of refined index of down-slope point
        :returns:
            - **q_peak**: q peak
            - **r_peak**: r peak
            - **s_peak**: s peak
        :rtype: tuple
        '''
        # logger.info('call qrspeaks()')

        left = left.copy()
        right = right.copy()

        r_value = np.zeros(left.size)
        s_value = np.zeros(left.size)
        q_value = np.zeros(left.size)
        r_loc = np.zeros(left.size, dtype=np.int)
        s_loc = np.zeros(left.size, dtype=np.int)
        q_loc = np.zeros(left.size, dtype=np.int)
        # reset minus index to 0 (caused by delay cancelling)
        left[np.nonzero(left < 0)] = 0
        right[np.nonzero(right < 0)] = 0

       ### Adaptive Threshold adjustment
        
        adjust = np.zeros(r_value.shape)
        
        ###
        

        if self.inv_r_peak_threshold is None or not self.inv_r_peak_locator_flag :
            for i in range(left.size):

                r_seg = x1[left[i]:right[i] + 1]
                r_value[i], r_loc[i] = np.max(r_seg), np.argmax(r_seg) + left[i]  # add offset
                    
                q_seg = x1[left[i]:r_loc[i] + 1]
                q_value[i], q_loc[i] = np.min(q_seg), np.argmin(q_seg) + left[i]  # add offset

                s_seg = x1[r_loc[i]:right[i] + 1]
                s_value[i], s_loc[i] = np.min(s_seg), np.argmin(s_seg) + r_loc[i]  # add offset

        else:

            for i in range(left.size):
                r_seg = x1[left[i]:right[i] + 1]
                
                r_arg_candidates = np.array([np.argmax(r_seg),np.argmax(np.abs(r_seg))])
                
                r_candidates = r_seg[r_arg_candidates]
                
                if r_candidates[:-1] != 0:
                    adjust[i] = np.abs(np.divide(r_candidates[1:],r_candidates[:-1]))
                else:
                    adjust[i] = np.Inf
                            
            
                
                if (adjust[i] < weighted(self.inv_r_peak_threshold,adjust[max(i-self.inv_r_peak_locator_window_size-1,0):i])).all():
                    if (np.sign(r_candidates) < 0).all():
                        r_value[i], r_loc[i] = r_seg[np.argmax(np.abs(r_seg))], np.argmax(np.abs(r_seg)) + left[i]  # add offset
                    else:
                        r_value[i], r_loc[i] = np.max(r_seg), np.argmax(r_seg) + left[i]  # add offset
                else:
                    r_value[i], r_loc[i] = r_seg[np.argmax(np.abs(r_seg))], np.argmax(np.abs(r_seg)) + left[i]  # add offset
                    
                q_seg = x1[left[i]:r_loc[i] + 1]
                s_seg = x1[r_loc[i]:right[i] + 1]

                if r_value[i] > 0:
                    q_value[i], q_loc[i] = np.min(q_seg), np.argmin(q_seg) + left[i]  # add offset
                    s_value[i], s_loc[i] = np.min(s_seg), np.argmin(s_seg) + r_loc[i]  # add offset

                else:
                    q_value[i], q_loc[i] = np.max(q_seg), np.argmax(q_seg) + left[i]  # add offset
                    s_value[i], s_loc[i] = np.max(s_seg), np.argmax(s_seg) + r_loc[i]  # add offset

        q_peak = Peak(q_loc, q_value, 'Q')
        r_peak = Peak(r_loc, r_value, 'R')
        s_peak = Peak(s_loc, s_value, 'S')

        return q_peak, r_peak, s_peak

    def _tpeak(self,x1, x2, q_peak, r_peak, s_peak, left, right, sample_rate):
        '''
        Detect T peak

        :param x1: original ECG signal
        :param x2: low pass filtered signal from ptompkins filter
        :param q_peak: q peak
        :type q_peak: Peak
        :param r_peak: r peak
        :type r_peak: Peak
        :param s_peak: s peak
        :type s_peak: Peak
        :param left: left array from poss regularity
        :param right: right array from poss regularity
        :returns: t_peak, width_t, t_inv, t_inv_rt_loc, t_loc_right
        :rtype: Peak, ndarray, ndarray, ndarray, ndarray
        '''
        t_loc = np.zeros(left.size, dtype=np.int)
        t_value = np.zeros(left.size)

        t_limit_left = np.zeros(left.size, dtype=np.int)
        t_limit_right = np.zeros(left.size, dtype=np.int)

        t_loc_left = np.zeros(left.size, dtype=np.int)
        t_loc_right = np.zeros(left.size, dtype=np.int)
        width_t = np.zeros(left.size, dtype=np.int)

        t_value_left = np.zeros(left.size)
        t_value_right = np.zeros(left.size)

        t_further_right = np.zeros(left.size, dtype=np.int)

        # T wave assumed to be not inverted
        t_inv = np.zeros(left.size, dtype=np.int)
        t_inv_rt_loc = np.zeros(left.size, dtype=np.int)

        # April 2020
        z_start = 1
        if self.conf is not None and self.conf.get('filter_close_qrs', False):
            z_start = 0

        for z in range(z_start, left.size - 1):
            t_inv[z] = 0
            # computing the length of the target window QRS_end[i]:QRS_begin[i+1]
            len_2 = q_peak.loc[z + 1] - right[z]
            len_1 = math.floor(0.68 * len_2)
            len_0 = math.floor(0.06 * len_2)

            # max value between QRS_end[i]:QRS_begin[i+1]
            xt_seg = x2[right[z] + len_0 + 6: right[z] + len_1 + 6 + 1]

            # initial t-loc: Get t_loc from x2 (delay by 6 due to the filter)
            t_loc[z] = np.argmax(xt_seg) + right[z] + len_0  # add offset
            t_value[z] = x1[t_loc[z]]

            # making further test to see if T is really upright
            # taking a point right of max i.e 300ms to the right of t_loc found
            # if exceed next Q peak, shorten to 200ms but within the lenght of x1
            right_max_t = t_loc[z] + math.floor(0.3 * len_2)
            if right_max_t > q_peak.loc[z + 1]:
                right_max_t = min(t_loc[z] + math.floor(0.2 * len_2), x1.size - 1)

            diff_T = abs(x1[right_max_t] - x1[t_loc[z]])
            if diff_T <= 0.03:
                t_inv[z] = 1

            # checking for the T wave inverted
            # if region to right of max is not flat then it is a peak
            # if abs(x1[right_max_t] - x1[t_loc[z]]) > 0.03:
            if t_inv[z] == 0:
                t_limit_left[z] = math.floor(t_loc[z] - 0.2 * sample_rate)
                t_limit_right[z] = math.floor(t_loc[z] + 0.2 * sample_rate)

                # to avoid segmentation fault
                t_limit_left[z] = min(max(t_limit_left[z], 0), x1.size - 1)
                t_limit_right[z] = min(t_limit_right[z], x1.size - 1)

                # finding T_begin and T_end from x1
                t_value_left[z], t_loc_left[z] = np.min(x1[t_limit_left[z]:t_loc[z] + 1]), np.argmin(
                    x1[t_limit_left[z]:t_loc[z] + 1]) + t_limit_left[z] + 1
                t_value_right[z], t_loc_right[z] = np.min(x1[t_loc[z]:t_limit_right[z] + 1]), np.argmin(
                    x1[t_loc[z]:t_limit_right[z] + 1]) + t_loc[z] + 1
                width_t[z] = abs(t_loc_right[z] - t_loc_left[z])
                # any point to right of right of T: ratio 0.3
                # if exceed next q peak, shorten to ratio 0.08
                t_further_right[z] = t_loc_right[z] + math.floor(0.3 * len_2)
                if t_further_right[z] > q_peak.loc[z + 1]:
                    t_further_right[z] = t_loc_right[z] + math.floor(0.08 * len_2)

                # to avoid segmentation fault
                t_further_right[z] = min(t_further_right[z], x1.size - 1)

                # checking if it is peak or left part of inverted T wave
                dif_tt = abs(x1[t_loc[z]] - x1[t_loc_right[z]])
                dif_t = abs(x1[t_further_right[z]] - x1[t_loc_right[z]])

                uplimit = max(r_peak.value[z], t_value[z])
                lowlimit = min(s_peak.value[z], q_peak.value[z])  # Average Samplitude

                if dif_t > 0.15 * abs(uplimit - lowlimit):  # wave rises again indicating it was a part of inverted T
                    t_inv[z] = 1
                if dif_tt <= 0.03:
                    t_inv[z] = 1

            else:
                t_inv[z] = 1

            if t_inv[z] == 1:

                # ++ matlab logic add
                t_value[z] = min(xt_seg)
                t_loc[z] = np.argmin(xt_seg)  # xt1
                t_loc[z] = t_loc[z] + right[z] + len_0
                t_value[z] = x1[t_loc[z]]

                if math.floor(t_loc[z] + (150 / 1000 * sample_rate)) > len(x1):
                    TT = len(x1)
                else:
                    TT = math.floor(t_loc[z] + (150 / 1000 * sample_rate))

                xt_right = x1[t_loc[z]:TT + 1]  # ??
                t_inv_rt_loc[z] = np.argmax(xt_right)
                t_inv_rt_loc[z] = t_inv_rt_loc[z] + t_loc[z]

        return Peak(t_loc, t_value, 'T'), width_t, t_inv, t_inv_rt_loc, t_loc_right

    def _ppeak(self,x1, x2, q_peak, r_peak, s_peak, t_peak, t_inv, t_inv_rt_loc, t_loc_right, left, right, sample_rate):
        '''
        Detect P peak

        :param x1: original ecg signal
        :param x2: low pass filtered ecg signal from tompkins filter
        :param q_peak: q peak
        :type q_peak: Peak
        :param r_peak: r peak
        :type r_peak: Peak
        :param s_peak: s peak
        :type s_peak: Peak
        :param t_peak: t peak
        :type t_peak: Peak
        :param t_inv: flag array of inverted T wave
        :param t_inv_rt_loc: index array of inverted T wave right end
        :param t_loc_right: index array of T wave right end
        :param left: left array of poss regularity
        :param right: right array of poss regularity
        :returns: p_peak, count_p, p_loc_left
        :rtype: Peak, ndarray, ndarray
        '''

        p_loc = np.zeros(left.size, dtype=np.int)
        p_value = np.zeros(left.size)

        p_loc_left = np.zeros(left.size, dtype=np.int)
        p_loc_right = np.zeros(left.size, dtype=np.int)

        p_limit_left = np.zeros(left.size, dtype=np.int)
        p_limit_right = np.zeros(left.size, dtype=np.int)

        baseline_amp = np.zeros(left.size)
        amp_p = np.zeros(left.size)
        width_p = np.zeros(left.size)
        count_p = np.ones(left.size, dtype=np.int)

        for z in range(left.size - 1):
            # 30% of the region taken for P estimation, Region 1, refer to paper Cardio24
            len_1 = math.floor(0.3 * (q_peak.loc[z + 1] - right[z]))
            # taking length from T to next Q, Region 2, refert to paper Cardio24
            len_2 = q_peak.loc[z + 1] - t_peak.loc[z]

            # taking whichever is min to find P from Region 1 and Region 2
            len_0 = min(len_1, len_2)
            if len_0 > 5:
                # QRS_begin[i+1] - (min(region1, region2)) : QRS_begin[i+1] , refer to paper Cardio24xp_seg = x2[q_peak.loc[z + 1] - len_0: q_peak.loc[z + 1]]
                p_loc[z] = np.argmax(xp_seg) + q_peak.loc[
                    z + 1] - len_0 - 6  # add offset, 6 is due to the delay of filter
                p_value[z] = x1[p_loc[z]]

                # finding P_begin and P_end
                p_limit_left[z] = max(math.floor(p_loc[z] - 0.055 * sample_rate), 0)
                p_limit_right[z] = min(math.floor(p_loc[z] + 0.055 * sample_rate), x1.size - 1)

                p_loc_left[z] = np.argmin(x1[p_limit_left[z]:p_loc[z] + 1]) + p_limit_left[z]
                p_loc_right[z] = np.argmin(x1[p_loc[z]:p_limit_right[z] + 1]) + p_loc[z]
                width_p[z] = abs(p_loc_right[z] - p_loc_left[z])

                p_loc[z] = np.argmax(x1[p_loc_left[z]:p_loc_right[z] + 1]) + p_loc_left[z]
                p_value[z] = x1[p_loc[z]]

                # baseline of p
                baseline_amp[z] = (x1[p_loc_left[z]] + x1[p_loc_right[z]]) / 2
                amp_p[z] = abs(p_value[z] - baseline_amp[z])

                if width_p[z] < 0.06 * sample_rate or amp_p[z] <= 0.0195:
                    count_p[z] = 0

                if t_inv[z] == 1:
                    # if left of P wave is very close to (10ms) right of inv T wave
                    if p_loc_left[z] - t_inv_rt_loc[z] <= 0.01 * sample_rate:
                        count_p[z] = 0
                # if P wave is very close to (70ms) T wave
                elif p_loc[z] - t_peak.loc[z] <= 0.07 * sample_rate:
                    count_p[z] = 0

        if len(count_p) > 0:
            count_p[-1] = 0

        return Peak(p_loc, p_value, 'P'), count_p, p_loc_left

    def _correct_peak(self,q_peak, r_peak, s_peak, left, right, x6, sample_rate):
        '''
        Refactoring the peaks
        To be used to refine the results of `qrspeaks(x1,left,right)`.

        :param q_peak: q peak
        :type q_peak: Peak
        :param r_peak: r peak
        :type r_peak: Peak
        :param s_peak: s peak
        :type s_peak: Peak
        :param left: left array of poss regularities
        :param right: right array of poss regularities
        :param x1: original ecg signal
        :param x6: x6_normalized (moving window integrated) output of tompkins filter
        :param sample_rate: sampling rate
        :type sample_rate: int
        '''

        q_loc = q_peak.loc.copy()
        q_value = q_peak.value.copy()

        r_loc = r_peak.loc.copy()
        r_value = r_peak.value.copy()

        s_loc = s_peak.loc.copy()
        s_value = s_peak.value.copy()

        # create a threshold based on the sig amplitude
        thres1 = 2.2 * np.mean(x6) * np.max(x6)
        thres1 = 0.3 if thres1 > 0.3 else thres1

        if r_peak.loc.size > 3:
            # Added to eliminate double R peak which has interval less than 0.2 second
            # April 2020
            if self.conf is not None and self.conf.get('filter_close_qrs', False):
                for j in np.nonzero(np.diff(r_loc) <= 0.2 * sample_rate)[0]:
                    offset_r = r_loc[j:j + 2] + 20
                    offset_r[offset_r >= x6.size] = x6.size - 1
                    # print(j, r_loc[j:j+2], x6[offset_r])

                    if x6[offset_r[0]] < x6[offset_r[1]]:
                        r_value[j] = np.nan
                    elif x6[offset_r[0]] > x6[offset_r[1]]:
                        r_value[j + 1] = np.nan

                q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right = self._remove_nan_value(
                    q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right
                )
            
            
            for j in np.nonzero(np.diff(r_loc) <= 0.31 * sample_rate)[0]:
                offset_r = r_loc[j:j + 2] + 20
                offset_r[offset_r >= x6.size] = x6.size - 1
                # print(j, r_loc[j:j+2], x6[offset_r])
                if x6[offset_r[0]] > thres1:
                    pass
                elif x6[offset_r[0]] < x6[offset_r[1]]:
                    r_value[j] = np.nan
                elif x6[offset_r[0]] > x6[offset_r[1]]:
                    r_value[j + 1] = np.nan

            q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right = self._remove_nan_value(
                q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right
            )

            threshold = 1.5 * np.mean(x6) * np.max(x6)

            # matlab findpeaks :
            # [qrspeaks, locs] = findpeaks(x6, fs, 'MinPeakHeight', thresh, 'MinPeakDistance', 0.5);
            # python way: use PeakUtils package
            # refer to: https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
            locs = peakutils.indexes(x6, thres=threshold, min_dist=0.5 * sample_rate)

            if locs.size < r_loc.size < locs.size * 3.5:
                r_peak_diff = np.diff(r_loc)
                has_narrow_rr_window = False
                # moving window size: 11 number of consective r wave, 10 RR diff
                # restore to orignal value if any of the moving window is narrow RR wave
                for j in range(r_peak_diff.size - 9):
                    if np.all(r_peak_diff[j:j + 10] < 0.4 * sample_rate):
                        has_narrow_rr_window = True
                        break

                if not has_narrow_rr_window:
                    for m in np.nonzero(np.logical_or(r_peak_diff <= 0.4 * sample_rate, r_peak_diff > 3 * sample_rate))[
                        0]:
                        offset_r = r_loc[m:m + 2] + 22
                        offset_r[offset_r >= x6.size] = x6.size - 1

                        if x6[offset_r[0]] >= thres1 and x6[offset_r[1]] >= thres1:
                            pass
                        elif x6[offset_r[0]] < x6[offset_r[1]]:
                            r_value[m] = np.nan
                        elif x6[offset_r[0]] > x6[offset_r[1]]:
                            r_value[m + 1] = np.nan

            q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right = self._remove_nan_value(
                q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right
            )

            # April 2020
            if self.conf is not None and self.conf.get('filter_close_qrs', False) and len(r_loc) > 0:
                if r_loc[0] <= 0.4*sample_rate:
                    r_value[0] = np.nan
                    q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right = self._remove_nan_value(
                        q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right
                    )

            for m in range(right.size - 1):
                right[m] = min(right[m], q_loc[m + 1])

        new_q_peak = Peak(q_loc, q_value, 'Q')
        new_r_peak = Peak(r_loc, r_value, 'R')
        new_s_peak = Peak(s_loc, s_value, 'S')

        return new_q_peak, new_r_peak, new_s_peak, left, right

    def _remove_nan_value(self,q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right):
        '''
        return new peaks vector by removing nan value in input peak vector

        output sequence is the same as the input sequence

        detect the index on nan based on r_value
        '''
        finite_ind = np.isfinite(r_value)

        r_value = r_value[finite_ind]
        q_value = q_value[finite_ind]
        s_value = s_value[finite_ind]
        r_loc = r_loc[finite_ind]
        q_loc = q_loc[finite_ind]
        s_loc = s_loc[finite_ind]

        left = left[finite_ind]
        right = right[finite_ind]
        return q_loc, q_value, r_loc, r_value, s_loc, s_value, left, right

def pan_tompkins_detect(sig, fs, components_to_detect='pqrst', conf=None):
    """
    Detect qrs and return a Peakmetrics object with the desired information

    :param sig: ndarray
    :param fs: float
    :param components_to_detect: string
    :param conf: dict
    :return: Peakmetrics
    """
    pan_tompkins = PanTompkins(sig=sig, fs=fs, conf=conf)
    pan_tompkins.detect(components_to_detect=components_to_detect)
    # return pan_tompkins.qrs_inds
    return pan_tompkins.peak_metrics

def pan_tompkins_detect_peaks(sig, fs, conf = None):
    return pan_tompkins_detect(sig,fs,components_to_detect='qrs', conf=conf).r.loc

def weighted(threshold,adjust_arr):
    ### TODO: Update this function for adaptive thresholding
    
    ### Temporary measure
    return threshold

# End of merged code from qrs.py

def ecg_filters(ecg_signal, sample_rate):
    '''
    Some ECG signal filters

    :param ecg_signal: ndarray, preprocessed ECG signal
    :param sample_rate: sampling rate of preprocessed ECG signal, default 200Hz
    '''

    # make sure to make the NaN and Inf values in the signal to finite;
    max_val= ecg_signal[np.isfinite(ecg_signal)].max()  # find the finite max and min
    min_val = ecg_signal[np.isfinite(ecg_signal)].min()

    ecg_signal[np.isinf(ecg_signal)] = max_val  # assign the non-finite values to finite extremes
    ecg_signal[np.isnan(ecg_signal)] = min_val

    # demean the signal
    ecg_signal = ecg_signal - np.mean(ecg_signal) #+
    # SG filter, Deprecated

    # Butter IIR filter
    freq_band = (0.5, 50)
    filtered_signal =  ecg_bandpass_filter(ecg_signal - np.mean(ecg_signal), freq_band[1],
                               freq_band[0], sample_rate)
    return filtered_signal

def read_ecg_mitdb(file='100'):
    """
    Read ECG data from MIT-BIH database
    param file: Record name e.g. '100'  
    return: (sig, hea, ann) tuple with signal, header, annotations
    """
    path = './database/mitdb/'+file
    sig, hea = wfdb.rdsamp(path)
    ann = wfdb.rdann(path, 'atr')
    return (sig, hea, ann)

def preprocess_ecg(sig, fs):
    """
    Preprocess ECG signal to detect P,Q,R,S,T waves
    param sig: Input ECG signal
    param fs: Sampling frequency
    return: (ecg, fs, morph, pqrst_locs_df) tuple with filtered ECG, 
            sampling freq, waveform morphology, P,Q,R,S,T locations DataFrame
    """
    ecg = ecg_filters(sig, fs)
    morph = get_ecg_metrics(ecg, fs)
    p_locs,q_locs,r_locs,s_locs,t_locs = morph.peak_metrics.p.loc, morph.peak_metrics.q.loc, \
                                    morph.peak_metrics.r.loc, morph.peak_metrics.s.loc, \
                                    morph.peak_metrics.t.loc
    key = ['p_locs', 'q_locs', 'r_locs', 's_locs', 't_locs'];
    value = [p_locs, q_locs, r_locs, s_locs, t_locs]
    pqrst_locs_df = pd.DataFrame(dict(zip(key, value)))
    return (ecg,fs,morph,pqrst_locs_df)

def split_avg_arr(arr, seg):
    """
    Split array into segments and calculate mean of each segment
    param arr: Input array
    param seg: Number of segments
    return: List of segment means
    """
    arr_split = np.array_split(arr, seg)
    arr_ave   = np.array([np.average(i) for i in arr_split])
    return arr_ave.tolist()

def read_obj_from_yaml(yaml_file):
    """
    Read object from YAML file
    param yaml_file: Path to YAML file
    return: Object with attributes from YAMLxp_seg = x2[q_peak.loc[z + 1] - len_0: q_peak.loc[z + 1]]
    """
    class obj(object):
        def __init__(self, d):
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, obj(b) if isinstance(b, dict) else b)
    with open(yaml_file, 'r') as stream:
        temp_yaml = yaml.safe_load(stream)
    return obj(temp_yaml)

def find_continue_int_in_array(arr):
    """
    Find continuous integer segments in array
    param arr: Input array
    return: List of continuous segments
    e.g. [1,4,5,6,10,15,16,17,18,22,25,26,27,28] 
      -> [[1], [4, 5, 6], [10], [15, 16, 17, 18], [22], [25, 26, 27, 28]]  
    """
    out = []
    for k, g in itertools.groupby( enumerate(arr), lambda x: x[1]-x[0] ) :
        out.append(list(map(itemgetter(1), g)))
    return out

def find_closest_segment(arr, inx):
    """
    Find segment in arr that is closest to index inx
    param arr: Input list of segments 
    param inx: Target index
    return: Closest segment to inx
    """
    out = []
    dist_s = []
    for a in arr:
        if len(a) > 3:
            dist = min(np.abs(inx - a[0]), np.abs(inx - a[1]))
            if dist < 3:
                return a 
            else:
                dist_s.append(dist)
    inx_tmp = np.argmin(dist_s) 
    out = arr[inx_tmp]
    return out    

def find_segment(ecg,fs,p_loc,q_loc,r_loc,s_loc,t_loc):
    """
    Extract P,QRS,T segments from ECG signal
    param ecg: Input ECG signal
    param fs: Sampling frequency 
    param p_loc, q_loc, r_loc, s_loc, t_loc: Locations of P,Q,R,S,T points
    return: (seg, p_wav, qrs_wav, t_wav) Tuple with full beat segment, P wave, QRS wave, T wave
    """
    p_wav   = ecg[p_loc-int(0.08*fs) : p_loc+int(0.08*fs)]
    qrs_wav = ecg[q_loc-int(0.05*fs) : s_loc+int(0.05*fs)]
    t_wav   = ecg[t_loc-int(0.09*fs) : t_loc+int(0.09*fs)]
    seg     = ecg[p_loc-int(0.08*fs) : t_loc+int(0.09*fs)]
    return (seg, p_wav, qrs_wav, t_wav)

def find_morph(ecg, fs, q_loc_this, r_loc_pre, r_loc_this, r_loc_pos, s_loc_this, t_loc_this):
    """
    Calculate morphology features for a single beat
    param ecg: Input ECG signal 
    param fs: Sampling frequency
    param q_loc_this, r_loc_pre, r_loc_this, r_loc_pos, s_loc_this, t_loc_this: 
          Locations of Q, previous R, current R, next R, S, T points
    return: (r_loc_pre_gap, r_loc_pos_gap, qs_loc_gap, r_val, t_val, q_val, t_area, r_area)
            Tuple with RR interval before, RR interval after, QS interval, R amp, T amp, 
            Q amp, T wave area, QRS wave area
    """
    r_loc_pre_gap = []; r_loc_pos_gap = []
    r_val = []; t_val = []; q_val = []
    t_wav = []; r_wav = []
    t_area = []; r_area = []

    r_loc_pre_gap = r_loc_this - r_loc_pre
    r_loc_pos_gap = r_loc_pos - r_loc_this
    qs_loc_gap = s_loc_this - q_loc_this
    r_val = ecg[r_loc_this]; t_val = ecg[t_loc_this]; q_val = ecg[q_loc_this]; 
    t_wav = ecg[t_loc_this-int(0.09*fs) : t_loc_this+int(0.09*fs)] 
    r_wav = ecg[q_loc_this-int(0.05*fs) : s_loc_this+int(0.05*fs)]
    t_area = np.trapz(t_wav); r_area = np.trapz(r_wav)
    return (r_loc_pre_gap, r_loc_pos_gap, qs_loc_gap, r_val, t_val, q_val, t_area, r_area)

def cal_feature(fs, qs_loc_gap, r_val, t_val, q_val, t_area):
    """
    Calculate PVC related features for a single beat
    param fs: Sampling frequency 
    param qs_loc_gap: QS interval
    param r_val, t_val, q_val: R, T, Q amplitudes
    param t_area: T wave area
    return: (c_qs_wid_irr, c_Qr_irr, c_t_inv, c_rT_irr) 
            Tuple with flags for irregular QS width, Q/R ratio, inverted T wave, R/T ratio
    """ 
    c_qs_wid_irr = 1 if (qs_loc_gap > 0.115 * fs) else 0
    c_Qr_irr = 1 if ((np.abs(q_val) * 0.3) > np.abs(r_val)) else 0
    c_t_inv = 1 if ((t_area < 0) or (t_val < 0)) else 0
    c_rT_irr = 1 if np.abs(t_val)  > 1.20* np.abs(r_val) else 0
    return (c_qs_wid_irr, c_Qr_irr, c_t_inv, c_rT_irr)

def get_morphs(peaks_info, max_seg=150):
    """
    Get morphology data for each beat
    param peaks_info: (ecg, fs, morph, pqrst_locs_df) Tuple 
    param max_seg: Maximum number of beats to analyze
    return: (temp_s, seg_s, morph_df) Tuple with feature templates, beat segments,
            morphology features DataFrame
    """
    ecg,fs,_,pqrst_locs_df = peaks_info
    p_locs,q_locs,r_locs,s_locs,t_locs = pqrst_locs_df['p_locs'], pqrst_locs_df['q_locs'],pqrst_locs_df['r_locs'],pqrst_locs_df['s_locs'],pqrst_locs_df['t_locs']

    temp_s = []; seg_s  = []; r_loc_s = [];
    r_loc_pre_gap_s = []; r_loc_pos_gap_s = []; qs_loc_gap_s = []; 
    r_val_s = []; t_val_s = []; q_val_s = []; 
    t_area_s = []; r_area_s = [];

    for i in range(1, len(r_locs)-1):
        seg, p_wav, qrs_wav, t_wav = find_segment(ecg,fs,p_locs[i],q_locs[i],r_locs[i],s_locs[i],t_locs[i])
        temp = np.array(split_avg_arr(p_wav,5) + split_avg_arr(qrs_wav,5) + split_avg_arr(t_wav,5))
        temp_s.append(temp.tolist()); seg_s.append(seg.tolist()); r_loc_s.append(r_locs[i])

        morph = find_morph(ecg, fs, q_locs[i], r_locs[i-1], r_locs[i], r_locs[i+1], s_locs[i], t_locs[i])
        (r_loc_pre_gap, r_loc_pos_gap, qs_loc_gap, r_val, t_val, q_val, t_area, r_area) = morph
        
        r_loc_pre_gap_s.append(r_loc_pre_gap); r_loc_pos_gap_s.append(r_loc_pos_gap); qs_loc_gap_s.append(qs_loc_gap);
        r_val_s.append(r_val); t_val_s.append(t_val); q_val_s.append(q_val);
        t_area_s.append(t_area); r_area_s.append(r_area)

    df_key = ['r_loc_s', 'r_loc_pre_gap_s', 'r_loc_pos_gap_s', 'qs_loc_gap_s', 'r_val_s', 't_val_s', 'q_val_s', 't_area_s', 'r_area_s']
    df_val = [r_loc_s, r_loc_pre_gap_s, r_loc_pos_gap_s, qs_loc_gap_s, r_val_s, t_val_s, q_val_s, t_area_s, r_area_s]
    morph_df = pd.DataFrame(dict(zip(df_key, df_val)))
            
    return (temp_s, seg_s, morph_df)

def get_features(fs, morphs_info, bases_info):
    """
    Get PVC related features for each beat

    param fs: Sampling frequency
    param morphs_info: (temp_s, seg_s, morph_df) Morphology info tuple
    param bases_info: (r_loc_s, preds, irrg_inx, base_inx, base_inx_seg)
                      Tuple with R locs, beat cluster assignments,
                      irregular beat indexes, baseline beat indexes,
                      continuous baseline beat segment indexes

    return: feats_df DataFrame with PVC features for each beat
    """
    temp_s, seg_s, morph_df = morphs_info
    r_loc_s, preds, irrg_inx, base_inx, base_inx_seg = bases_info

    r_loc_s = morph_df['r_loc_s']
    r_loc_pre_gap_s = morph_df['r_loc_pre_gap_s']
    r_loc_pos_gap_s = morph_df['r_loc_pos_gap_s']
    qs_loc_gap_s = morph_df['qs_loc_gap_s']
    r_val_s = morph_df['r_val_s']
    t_val_s = morph_df['t_val_s']
    q_val_s = morph_df['q_val_s']
    t_area_s = morph_df['t_area_s']
    r_area_s = morph_df['r_area_s']

    r_locs = np.array(r_loc_s)
    r_loc_pre_gaps = np.array(r_loc_pre_gap_s)
    r_loc_pos_gaps = np.array(r_loc_pos_gap_s)
    qs_loc_gaps = np.array(qs_loc_gap_s)
    r_vals = np.array(r_val_s)
    t_vals = np.array(t_val_s)
    q_vals = np.array(q_val_s)
    t_areas = np.array(t_area_s)

    c_qs_wid_irr_s = []
    c_Qr_irr_s = []
    c_t_inv_s = []
    c_rT_irr_s = []
    c_r_loc_pre_irr_s = []
    c_r_loc_pos_irr_s = []
    c_r_val_irr_sma_s = []
    c_r_val_irr_big_s = []

    for i in range(len(r_loc_s)):
        r_inxs_reg = find_closest_segment(base_inx_seg, i)
        r_locs_reg = r_locs[r_inxs_reg]
        r_vals_reg = r_vals[r_inxs_reg]
        q_vals_reg = q_vals[r_inxs_reg]

        c_qs_wid_irr, c_Qr_irr, c_t_inv, c_rT_irr = cal_feature(fs, qs_loc_gaps[i], r_vals[i],
                                                                t_vals[i], q_vals[i], t_areas[i])

        c_r_loc_pre_irr = int(1.20*r_loc_pre_gaps[i] < np.mean(r_locs_reg[1:] - r_locs_reg[:-1]))
        c_r_loc_pos_irr = int(r_loc_pos_gaps[i] > 1.20*np.mean(r_locs_reg[1:] - r_locs_reg[:-1]))
        c_r_val_irr_sma = int(1.30*np.abs(r_vals[i]) < np.abs(np.mean(r_vals_reg)))
        c_r_val_irr_big = int(np.abs(r_vals[i]) > 1.30*np.abs(np.mean(r_vals_reg)))

        c_qs_wid_irr_s.append(c_qs_wid_irr)
        c_Qr_irr_s.append(c_Qr_irr)
        c_t_inv_s.append(c_t_inv)
        c_rT_irr_s.append(c_rT_irr)
        c_r_loc_pre_irr_s.append(c_r_loc_pre_irr)
        c_r_loc_pos_irr_s.append(c_r_loc_pos_irr)
        c_r_val_irr_sma_s.append(c_r_val_irr_sma)
        c_r_val_irr_big_s.append(c_r_val_irr_big)

    cols = ['r_loc_s', 'c_qs_wid_irr_s', 'c_Qr_irr_s', 'c_t_inv_s', 'c_rT_irr_s',
            'c_r_loc_pre_irr_s', 'c_r_loc_pos_irr_s', 'c_r_val_irr_sma_s', 'c_r_val_irr_big_s']

    data = [r_loc_s, c_qs_wid_irr_s, c_Qr_irr_s, c_t_inv_s, c_rT_irr_s,
c_r_loc_pre_irr_s, c_r_loc_pos_irr_s, c_r_val_irr_sma_s, c_r_val_irr_big_s]

    feats_df = pd.DataFrame(dict(zip(cols, data)))

    return feats_df


def cal_cluster(X, eps):
    """
    Perform DBSCAN clustering

    param X: Input feature data
    param eps: Epsilon distance threshold

    return: (predicts, n_predicts)
            Tuple with cluster assignments and number of clusters
    """
    dbscan = cluster.DBSCAN(eps=eps)
    predicts = dbscan.fit_predict(X)
    n_predicts = len(set(predicts)) - (1 if -1 in predicts else 0)

    return predicts, n_predicts


def find_majority(morphs_info, eps_base, n_cluster=(2,4), n_grid=[0.1,0.3,0.6,1,3,6,10]):
    """
    Find majority beats via clustering

    param morphs_info: (temp_s, seg_s, morph_df) Morphology info tuple
    param eps_base: Base epsilon threshold
    param n_cluster: (min,max) valid number of clusters
    param n_grid: Grid of epsilon factors

    return: (preds_o, irrg_inx_o, base_inx_o, base_inx_seg_o)
            Tuple with optimal cluster assignments,
            irregular beat indexes, baseline beat indexes,
            continuous baseline beat segment indexes
    """
    temp_s, _, morph_df = morphs_info

    n_baseline_s = []
    preds_s = []

    for rate in n_grid:
        preds, n_pred = cal_cluster(temp_s, eps=rate*eps_base)

        if n_cluster[0] <= n_pred <= n_cluster[1]:
            n_baseline = (preds == np.median(preds)).sum()
            n_baseline_s.append(n_baseline)
            preds_s.append(preds)

    inx_max = np.argmax(n_baseline_s)
    preds_o = preds_s[inx_max]

    irrg_inx_o = np.where(preds_o != np.median(preds_o))[0]
    base_inx_o = np.where(preds_o == np.median(preds_o))[0]
    base_inx_seg_o = find_continue_int_in_array(base_inx_o.tolist())

    return preds_o, irrg_inx_o, base_inx_o, base_inx_seg_o


class PROCESS:
    """
    Class to process ECG and find baseline rhythm
    """

    def __init__(self, ecg, fs):
        """
        Initialize with ECG signal and sampling frequency
        """
        self.ecg = ecg
        self.fs = fs
        self.peaks_info = preprocess_ecg(ecg, fs)
        self.morphs_info = get_morphs(self.peaks_info)
        self.bases_info = self.find_base_rhythm()

    def find_base_rhythm(self):
        """
        Find baseline rhythm via clustering

        return: (r_loc_s, preds_o, irrg_inx_o, base_inx_o, base_inx_seg_o)
                Tuple with R locs, optimal cluster assignments,
                irregular beat indexes, baseline beat indexes,
                continuous baseline beat segment indexes
        """
        eps_base = 0.5 * np.median(np.abs(self.ecg))
        preds_o, irrg_inx_o, base_inx_o, base_inx_seg_o = find_majority(self.morphs_info, eps_base)
        r_loc_s = self.morphs_info[2]['r_loc_s']

        return r_loc_s, preds_o, irrg_inx_o, base_inx_o, base_inx_seg_o


class DETECT(PROCESS):
    """
    Class to detect PVCs
    Inherits from PROCESS class
    """

    def __init__(self, ecg, fs):
        """
        Initialize with ECG signal and sampling frequency
        """
        super().__init__(ecg, fs)
        self.feats_info = get_features(self.fs, self.morphs_info, self.bases_info)


class TEST:
    """
    Class with test methods
    """

    @staticmethod
    def test1():
        """
        Test finding baseline rhythm
        """
        sig, _, ann = read_ecg_mitdb('109')
        ecg = sig[:,0]
        fs = ann.fs

        process = PROCESS(ecg, fs)
        r_loc_s, preds_o, irrg_inx_o, base_inx_o, base_inx_seg_o = process.bases_info

        plt.figure(figsize=(10,4))
        plt.plot(ecg)
        plt.plot(r_loc_s, ecg[r_loc_s], 'o', markersize=5, label='All Beats')
        plt.plot(r_loc_s[base_inx_seg_o[0]], ecg[r_loc_s[base_inx_seg_o[0]]], 's', markersize=8, label='Baseline Beats')
        plt.plot(r_loc_s[irrg_inx_o], ecg[r_loc_s[irrg_inx_o]], '*', markersize=10, label='Irregular Beats')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def test2():
        """
        Test detecting PVCs
        """
        sig, _, ann = read_ecg_mitdb('109')
        ecg = sig[:,0]
        fs = ann.fs

        v_detector = DETECT(ecg, fs)
        feats_df = v_detector.feats_info

        print("PVC Features:")
        print(feats_df.head())


if __name__ == "__main__":
    TEST.test1()
    TEST.test2()
