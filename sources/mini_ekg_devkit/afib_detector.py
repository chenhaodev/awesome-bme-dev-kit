import numpy as np
import math
from scipy import signal
import resampy
from pyentrp import entropy as ent

class AFibDetector:
    def __init__(self):
        self.low_horz_time = 0.0385
        self.high_horz_time = 0.14
        self.low_vert_thr = 1.0
        self.high_vert_thr = 5.26
        self.range_horz_l = 0.35
        self.range_horz_r = 0
        self.scale = 3
        self.peak_pos = []
        self.sig = np.array([])
        self.fso = 360
        self.gain = 200
        self.val_conv_segment = []

    def detect_afib(self, sig, sampling_rate, r_peak_pos, gain_adu):
        if sampling_rate != 360:
            self.sig = resampy.resample(sig, sampling_rate, 360)
        else:
            self.sig = sig

        self.peak_pos = r_peak_pos
        self.gain = gain_adu
        self.fso = 360

        p_idx = self.detect_p_waves()
        rr_intervals = np.diff(r_peak_pos)
        rr_intervals = remove_pause(rr_intervals)
        cosen_val = cosen_slide(rr_intervals)
        sigma_val = sigma_slide(rr_intervals)

        if cosen_val > 0.8 and sigma_val < 0.5:
            return True
        else:
            return False

    def detect_p_waves(self):
        self.val_conv_segment = []
        segment, strt_idx = self.ext_seg(self.sig, self.peak_pos, self.fso)
        idx_cp, val_cp = self.set_conv_w(segment, self.scale, self.fso, 1. - self.range_horz_l, self.range_horz_r)

        p_idx = []
        for i in range(len(idk)):
            if len(idk[i]) > 0:
                p_idx.append(strt_idx[i] + self.average_list(idk[i]))

        return p_idx

    def ext_seg(self, signal, qrs_ps, fso):
        xr = np.ceil(qrs_ps * fso)
        str_idx = []
        out_segment = []
        for i in range(xr.size - 1):
            if signal.size > math.ceil(xr[i + 1]):
                segment = signal[math.ceil(xr[i]) - 1:math.ceil(xr[i + 1])]
                out_segment.append(segment)
                str_idx.append(math.ceil(xr[i]))
        return out_segment, str_idx

    def set_conv_w(self, segment, scale, sampling_rate, range_left, range_right):
        idk_cp = []
        val_cp = []
        for p_wave in segment:
            if p_wave.size > 0:
                val, index_com = self.wave_p_conv(p_wave, scale, 1 / sampling_rate)
                new_wvt = self.rm_q_right(val, p_wave.size, index_com, range_right)

                self.val_conv_segment.append(new_wvt)

                peak_pnt, val_peak_pnt = self.find_peaks(new_wvt, p_wave.size, range_left)

                idk_cp.append(peak_pnt)
                val_cp.append(val_peak_pnt)

        return idk_cp, val_cp

    def wave_p_conv(self, pwave, scale, step):
        time = np.arange(-0.1, 0.1, step)
        time = np.append(time, 0.1)
        gauss_let = self.firgaslet(time, scale)
        gen_wave = np.ones(gauss_let.size) * pwave[0]
        gen_wave = np.append(gen_wave, pwave)
        gen_wave = np.append(gen_wave, np.ones(gauss_let.size) * pwave[0])

        val = []
        index_com = []
        for index in range(pwave.size + gauss_let.size):
            index_com.append(-np.floor(gauss_let.size / 2) + 1 + index + 1)
            sum_out = np.dot(gauss_let, gen_wave[index:gauss_let.size + index])
            val.append(sum_out)
        return val, index_com

    def firgaslet(self, t, scale=3, sigma=0.0025):
        t = t / (2 ** scale)
        out = 1
        out = -(np.sqrt(2) / (np.pi ** (1 / 4) * sigma ** (1 / 2))) * out
        out = out * t
        out = out * np.exp(-t ** 2 / (2 * sigma ** 2))
        out = out / (2 ** scale)
        return out

    def rm_q_right(self, wvt, len_seg, index_com, range_right):
        idk_one = [i for i in range(len(index_com)) if index_com[i] == 1][0]
        idk_last = [i for i in range(len(index_com)) if index_com[i] == len_seg][0]
        per5 = math.ceil(len_seg * range_right)

        new_wvt = wvt[idk_one:idk_last - per5 + 1]
        return new_wvt

    def find_peaks(self, new_wvt, p_wave_size, range_left):
        peaks, _ = signal.find_peaks(new_wvt)
        val_inv = 1.01 * np.max(new_wvt) - new_wvt
        peaks_inv, _ = signal.find_peaks(val_inv)

        peak_pnt = np.sort(np.append(peaks, peaks_inv))

        len_cons = math.ceil(range_left * p_wave_size)
        ind_peak_pnt = np.where(peak_pnt > len_cons)
        peak_pnt = peak_pnt[tuple(ind_peak_pnt)]

        if len(peak_pnt) > 0:
            val_peak_pnt = [new_wvt[i] for i in np.nditer(peak_pnt)]
        else:
            val_peak_pnt = []

        peak_pnt = np.array(peak_pnt)
        val_peak_pnt = np.array(val_peak_pnt)

        return peak_pnt, val_peak_pnt

    def average_list(self, list_inp):
        out = sum(list_inp) / float(len(list_inp))
        out = round(out)
        return out

def remove_pause(RRs, rthr=3.0):
    return RRs[RRs < rthr]

def cosen_slide(RRs, slid_win=5, w=1, r=0.03):
    cose = -10000
    if RRs.size >= slid_win:
        for i in range(RRs.size - slid_win + 1):
            cose = max(cose, cosen(RRs[i:i + slid_win], w, r))
    else:
        cose = max(cose, cosen(RRs, w, r))

    return cose

def cosen(seq, w=1, r=0.03):
    sample_entropy = ent.sample_entropy(seq, w, r)
    sp_en = sample_entropy[w - 1]

    cosen_val = sp_en + np.log(2. * r) - np.log(np.mean(seq))

    if np.isnan(cosen_val):
        return 0
    elif np.isinf(cosen_val):
        return 1
    else:
        return cosen_val

def sigma_slide(RRs, cutoff=0.14):
    if RRs.size <= 2:
        return 1

    std_arr = np.zeros(RRs.size - 2)
    for i in range(RRs.size - 2):
        arr = np.array([RRs[i], RRs[i + 1], RRs[i + 2]])
        std_diff = np.std(np.diff(arr), ddof=1)
        mean_arr = np.mean(arr)
        std_arr[i] = std_diff / mean_arr

    less_std = std_arr[std_arr < cutoff]
    less_std = less_std.size

    out = less_std / std_arr.size
    return out