import numpy as np
import peakutils
from scipy.signal import lfilter

class PanTompkins:
    def __init__(self, sig, fs, conf=None):
        self.sig = sig
        self.fs = fs
        self.conf = conf
        self.inv_r_peak_locator_flag = conf.get("inv_r_peak_flag", False) if conf else False
        self.inv_r_peak_threshold = conf.get("inv_r_peak_locator_threshold", 2.5) if conf else 2.5
        self.inv_r_peak_locator_window_size = conf.get("inv_r_peak_locator_window_size", 1) if conf else 1

        self.pan_tompkins_filter = conf.get("pan_tompkins_filter", "legacy") if conf else "legacy"
        self._filter_ptompkins = self.choose_filter(self.pan_tompkins_filter)

    def choose_filter(self, filter_option):
        if filter_option == "legacy":
            return self._filter_ptompkins_legacy
        elif filter_option == "modified":
            return self._filter_ptompkins_modified
        else:
            raise ValueError('Filter choice must be one of ["legacy", "modified", None]')

    def detect_r_peaks(self):
        self._filter_ptompkins()
        threshold = ((np.mean(self.x6) + np.median(self.x6)) / 2) * max(self.x6)
        poss_reg = (self.x6 > threshold).astype(int)
        left, right = self._get_up_down_slope(poss_reg)

        for m in range(left.size):
            seg_x6 = self.x6[left[m]:right[m] + 1]
            seg_x6_max = max(seg_x6)
            seg_threshold = 0.5 * seg_x6_max if seg_x6_max < 0.55 else 0.35 * seg_x6_max

            seg_poss = (seg_x6 > seg_threshold).astype(int)
            additional_left, additional_right = self._get_up_down_slope(seg_poss)

            if additional_left.size == additional_right.size == 1:
                left[m], right[m] = left[m] + additional_left, left[m] + additional_right

        if self.pan_tompkins_filter == "modified":
            left -= (6 + 16 + 2)
            right -= (6 + 16 + 2)
        else:
            left -= (6 + 16)
            right -= (6 + 16)

        _, r_peak, _ = self._qrspeaks(self.x1, left, right)
        return r_peak.loc

    def _filter_ptompkins_legacy(self):
        self.x1 = (self.sig - np.mean(self.sig)) / np.max(np.abs(self.sig - np.mean(self.sig)))

        b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
        a = [1, -2, 1]
        h_lp = lfilter(b, a, np.concatenate(([1], np.zeros(12))))
        self.x2 = np.convolve(self.x1, h_lp)

        b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32,
             -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        a = [1, -1]
        h_hp = lfilter(b, a, np.concatenate(([1], np.zeros(32))))
        self.x3 = np.convolve(self.x2, h_hp)

        h = np.array([-1, -2, 0, 2, 1]) / 8
        delay = 2
        self.x4 = np.convolve(self.x3, h)[delay:delay + self.sig.size]

        self.x5 = self.x4 ** 2

        h = np.ones(31) / 31
        delay = 15
        self.x6 = np.convolve(self.x5, h)[delay:delay + self.sig.size]

    def _filter_ptompkins_modified(self):
        self.x1 = (self.sig - np.mean(self.sig)) / np.max(np.abs(self.sig - np.mean(self.sig)))

        b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
        a = [1, -2, 1]
        h_lp = lfilter(b, a, np.concatenate(([1], np.zeros(12))))
        self.x2 = np.convolve(self.x1, h_lp)

        b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        a = [1, 1]
        h_hp = lfilter(b, a, np.concatenate(([1], np.zeros(32))))
        self.x3 = np.convolve(self.x2, h_hp)

        b = np.array([1, 2, 0, -2, -1]) / 8
        a = [1]
        delay = 2
        slope_fil = lfilter(b, a, np.concatenate(([1], np.zeros(4))))
        self.x4 = np.convolve(self.x3, slope_fil)[delay:delay + self.sig.size]

        self.x5 = self.x4 ** 2

        h = np.ones(31) / 31
        delay = 15
        self.x6 = np.convolve(self.x5, h)[delay:delay + self.sig.size]

    def _get_up_down_slope(self, poss_signal):
        up_slope = np.nonzero(np.diff(np.concatenate(([0], poss_signal))) == 1)[0]
        down_slope = np.nonzero(np.diff(np.concatenate((poss_signal, [0]))) == -1)[0]
        return up_slope, down_slope

    def _qrspeaks(self, x1, left, right):
        left = left.copy()
        right = right.copy()

        r_value = np.zeros(left.size)
        r_loc = np.zeros(left.size, dtype=np.int)

        left[np.nonzero(left < 0)] = 0
        right[np.nonzero(right < 0)] = 0

        adjust = np.zeros(r_value.shape)

        if self.inv_r_peak_threshold is None or not self.inv_r_peak_locator_flag:
            for i in range(left.size):
                r_seg = x1[left[i]:right[i] + 1]
                r_value[i], r_loc[i] = np.max(r_seg), np.argmax(r_seg) + left[i]
        else:
            for i in range(left.size):
                r_seg = x1[left[i]:right[i] + 1]

                r_arg_candidates = np.array([np.argmax(r_seg), np.argmax(np.abs(r_seg))])
                r_candidates = r_seg[r_arg_candidates]

                if r_candidates[:-1] != 0:
                    adjust[i] = np.abs(np.divide(r_candidates[1:], r_candidates[:-1]))
                else:
                    adjust[i] = np.Inf

                if (adjust[i] < self.inv_r_peak_threshold).all():
                    if (np.sign(r_candidates) < 0).all():
                        r_value[i], r_loc[i] = r_seg[np.argmax(np.abs(r_seg))], np.argmax(np.abs(r_seg)) + left[i]
                    else:
                        r_value[i], r_loc[i] = np.max(r_seg), np.argmax(r_seg) + left[i]
                else:
                    r_value[i], r_loc[i] = r_seg[np.argmax(np.abs(r_seg))], np.argmax(np.abs(r_seg)) + left[i]

        r_peak = Peak(r_loc, r_value, 'R')
        return None, r_peak, None

def detect_r_peaks(sig, fs, conf=None):
    pan_tompkins = PanTompkins(sig=sig, fs=fs, conf=conf)
    return pan_tompkins.detect_r_peaks()

###usage###
#r_peak_indices = detect_r_peaks(ecg_signal, sampling_frequency, conf=config_dict)
