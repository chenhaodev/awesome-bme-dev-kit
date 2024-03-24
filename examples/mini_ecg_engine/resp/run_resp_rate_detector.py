import pandas as pd
import numpy as np
import os
from resp_rate_detector import estimate_resp_rate
import glob
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import mean_absolute_error
import boto3
from os.path import expanduser
import h5py
from scipy.interpolate import interp1d
import gc


def run_bidmc():
    """
    Run the respiration rate estimation on the BIDMC dataset.
    """
    src_dir = "/Users/fan.c/data/respiration_rate_data/bidmc-ppg-and-respiration-dataset-1.0.0/"
    signals_files = sorted(glob.glob(src_dir + "bidmc_csv/*_Signals.csv"))
    numerics_files = sorted(glob.glob(src_dir + "bidmc_csv/*_Numerics.csv"))

    config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bidmc.yaml')))

    mae_dataframe = pd.DataFrame(columns=["filename"] + config['signals_to_use'] + ["all"])
    mae_dataframe.set_index("filename", inplace=True)
    for signal_file, numerics_file in zip(signals_files, numerics_files):
        print(signal_file)
        filename = os.path.split(signal_file)[1][:8]
        print(filename)

        if filename == filename:
            raw_data = pd.read_csv(signal_file)
            ecg_data = raw_data[["Time [s]", " II"]]
            rr_metrics, plt_fig, chan_qualities = estimate_resp_rate(ecg_data.iloc[:,0].values,
                                                                     ecg_data.iloc[:,1].values,
                                                                     125,
                                                                     plot_fig=True,
                                                                     config=config)

            rr_metrics.to_csv(
                src_dir + "result_csv/%s_rr_metrics.csv" % filename, index=True
            )

            plt_fig.savefig(src_dir + "figures/%s_rr_q_metric.png" % filename)

            numerics_data = pd.read_csv(numerics_file)
            numerics_data.set_index("Time [s]", inplace=True)

            ax = rr_metrics.plot(y="RR")
            numerics_data.plot(y=" RESP", ax=ax)
            ax.set_ylabel("Respiration Rate")
            ax.set_ylim([6, 42])
            ax.legend(["Estimated", "Actual"])
            plt.savefig(src_dir + "figures/%s_estimated_actual_rr.png" % filename)
            plt.close("all")
            chan_qualities['Device'] = numerics_data[' RESP']
            rr_comparison = pd.concat(
                [pd.Series(numerics_data[" RESP"]).rename("Device")] +
                [pd.Series(rr_metrics[col]) for col in rr_metrics.columns if col.startswith('RR')],
                ignore_index=False,
                axis=1,
            )
            chan_qualities.to_csv(
                src_dir + "result_csv/%s_qualities.csv" % filename, index=True

            )
            rr_comparison.to_csv(
                src_dir + "result_csv/%s_rr_comparison.csv" % filename, index=True
            )

            rr_comparison.dropna(inplace=True)
            rr_comparison = rr_comparison[~(rr_comparison == 0).any(axis=1)]

            try:
                curr_mae = [mean_absolute_error(rr_comparison.Device, rr_comparison[col])
                            for col in rr_comparison.columns if col.startswith('RR')]
                mae_dataframe.loc[filename] = curr_mae
            except:
                pass

    mae_dataframe.to_csv(src_dir + "result_csv/mae_comparison.csv", index=True)


def run_vitalconnect_s3():
    """
    Run the respiration rate estimation on the VitalConnect S3 dataset.
    """
    BUCKET_NAME = 'biofourmis-research-data-ap-southeast-1'
    ecg_folder = 'Home Hospital data/ecg'
    vital_folder = 'Home Hospital data/vitals/'

    src_dir = expanduser('~/results/home_hospital_data/')

    s3c = boto3.client('s3', region_name='ap-southeast-1')
    ecg_files = [x['Key'] for x in s3c.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=ecg_folder,
        FetchOwner=False,
        RequestPayer='requester'
    )['Contents'][1:]]
    vitals_files = [x['Key'] for x in s3c.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=vital_folder,
        FetchOwner=False,
        RequestPayer='requester'
    )['Contents'][1:]]

    common_files = set([x.split('/')[-1].split('_')[0] for x in ecg_files]).intersection(set(
        [x.split('/')[-1].split('_')[0] for x in vitals_files]
    ))
    common_files = sorted(list(common_files))
    config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vitalconnect.yaml')))

    mae_dataframe = pd.DataFrame(columns=["filename"] + config['signals_to_use'] + ["all"])
    mae_dataframe.set_index("filename", inplace=True)
    ecg_fs = 125
    print(common_files)
    for key in common_files:
        print(f'Home Hospital data/ecg/{key}_ecg.csv')
        try:
            raw_data = pd.read_csv(s3c.get_object(Bucket=BUCKET_NAME,
                                                  Key=f'Home Hospital data/ecg/{key}_ecg.csv')['Body'],
                                   encoding='utf-8')
            raw_data["Time"] = pd.to_datetime(raw_data["Time"], unit="ms").dt.round("s")
            ecg_data = raw_data[["Time", "ec"]]
            ecg_data.set_index("Time", inplace=True)
            ecg_data = ecg_data.loc[sorted(ecg_data.index)]
            print('data loaded')

            vital_data = pd.read_csv(s3c.get_object(Bucket=BUCKET_NAME,
                                                    Key=f'Home Hospital data/vitals/{key}_vitals.csv')['Body'],
                                     encoding='utf-8')
            vital_data["Time"] = pd.to_datetime(vital_data["Time"], unit="ms").dt.round("s")
            vital_data.set_index("Time", inplace=True)
        except KeyError:
            continue
        if vital_data.shape[0] < 1:
            continue

        start_time = vital_data.index[1]
        end_time = vital_data.index[-1]

        ecg_data = ecg_data[start_time:end_time]
        ecg_data.reset_index(inplace=True)

        vital_data = vital_data[start_time:end_time]

        rr_metrics, plt_fig, chan_qualities = estimate_resp_rate(ecg_data.iloc[:, 0].values,
                                                                 ecg_data.iloc[:, 1].values,
                                                                 ecg_fs,
                                                                 plot_fig=True,
                                                                 config=config)

        rr_metrics.to_csv(
            src_dir + "result_csv/%s_rr_metrics.csv" % (key), index=True
        )

        plt_fig.savefig(src_dir + "figures/%s_rr_q_metric.png" % (key))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if rr_metrics.shape[0] > 0:
            rr_metrics.plot(y="RR", ax=ax)
            vital_data.plot(y="re", ax=ax)
        ax.set_ylabel("Respiration Rate")
        ax.set_ylim([1, 60])
        ax.legend(["Estimated", "Actual"])
        plt.savefig(src_dir + "figures/%s_estimated_actual_rr.png" % (key))
        plt.close("all")

        rr_comparison = pd.concat(
            [pd.Series(vital_data["re"]).rename("Device")] +
            [pd.Series(rr_metrics[col]) for col in rr_metrics.columns if col.startswith('RR')],
            ignore_index=False,
            axis=1,
        )

        rr_comparison.to_csv(
            src_dir + "result_csv/%s_rr_comparison.csv" % (key), index=True
        )

        rr_comparison.dropna(inplace=True)
        rr_comparison = rr_comparison[~(rr_comparison == 0).any(axis=1)]

        try:
            curr_mae = [mean_absolute_error(rr_comparison.Device, rr_comparison[col])
                        for col in rr_comparison.columns if col.startswith('RR')]
            mae_dataframe.loc[key] = curr_mae
        except:
            pass
        del ecg_data
        del vital_data
        gc.collect()

    mae_dataframe.to_csv(src_dir + "result_csv/mae_comparison.csv", index=True)


def _get_refs(a, kernel_size, stride_length):
    """
    Get reference respiration rates from the labeled data.

    :param a: Labeled data dictionary
    :param kernel_size: Kernel size for calculating respiration rates
    :param stride_length: Stride length for sliding window
    :return: Reference times and respiration rates
    """
    temp = a['labels']['co2']['startinsp']['x'][:].reshape(-1) / a['param']['samplingrate']['ecg'][0][0]
    temp1 = a['labels']['co2']['startexp']['x'][:].reshape(-1) / a['param']['samplingrate']['ecg'][0][0]
    insp_rates = []
    exp_rates = []
    times = []
    for t in np.arange(kernel_size, 481, stride_length):
        insp_rates.append(60 / np.diff(temp[(temp > t - kernel_size) & (temp < t)]).mean())
        exp_rates.append(60 / np.diff(temp1[(temp1 > t - kernel_size) & (temp1 < t)]).mean())
        times.append(t)
    rates = 0.5 * (np.array(insp_rates) + np.array(exp_rates))
    return times, rates


def run_capno():
    """
    Run the respiration rate estimation on the Capnobase dataset.
    """
    src_dir = expanduser('~/data/respiration_rate_data/TBME2013-PPGRR-Benchmark_R3/')
    signals_files = sorted(glob.glob(src_dir + "data/*_8min.mat"))

    config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bidmc.yaml')))
    print(config)
    mae_dataframe = pd.DataFrame(columns=["filename"] + config['signals_to_use'] + ["all"])
    mae_dataframe.set_index("filename", inplace=True)
    for signal_file in signals_files:
        print(signal_file)
        filename = os.path.split(signal_file)[1][:8]
        print(filename)
        raw_data = h5py.File(signal_file, 'r')
        ref_time, refs = _get_refs(raw_data, config['kernel_size'], config['stride_length'])

        ecg_fs = int(raw_data['param']['samplingrate']['ecg'][0][0])
        ecg_time = np.arange(0, len(raw_data['signal']['ecg']['y'][:].reshape(-1)) / ecg_fs, 1 / ecg_fs)
        ecg_data = pd.DataFrame({'Time': ecg_time, 'ecg': raw_data['signal']['ecg']['y'][:].reshape(-1)})
        rr_metrics, plt_fig, chan_qualities = estimate_resp_rate(ecg_data.iloc[:, 0].values,
                                                                 ecg_data.iloc[:, 1].values,
                                                                 ecg_fs,
                                                                 plot_fig=True,
                                                                 config=config)

        rr_metrics.to_csv(
            src_dir + "result_csv/%s_rr_metrics.csv" % filename, index=True
        )

        plt_fig.savefig(src_dir + "figures/%s_rr_q_metric.png" % filename)

        ax = rr_metrics.plot(y="RR")
        ax.plot(ref_time, refs)
        ax.set_ylabel("Respiration Rate")
        ax.set_ylim([6, 42])
        ax.legend(["Estimated", "Actual"])
        plt.savefig(src_dir + "figures/%s_estimated_actual_rr.png" % filename)
        plt.close("all")
        chan_qualities['Device'] = refs
        rr_comparison = pd.concat(
            [pd.Series(refs).rename("Device")] +
            [pd.Series(rr_metrics[col]) for col in rr_metrics.columns if col.startswith('RR')],
            ignore_index=False,
            axis=1,
        )
        chan_qualities.to_csv(
            src_dir + "result_csv/%s_qualities.csv" % filename, index=True

        )
        rr_comparison.to_csv(
            src_dir + "result_csv/%s_rr_comparison.csv" % filename, index=True
        )

        rr_comparison.dropna(inplace=True)
        rr_comparison = rr_comparison[~(rr_comparison == 0).any(axis=1)]

        try:
            curr_mae = [mean_absolute_error(rr_comparison.Device, rr_comparison[col])
                        for col in rr_comparison.columns if col.startswith('RR')]
            mae_dataframe.loc[filename] = curr_mae
        except:
            pass

    mae_dataframe.to_csv(src_dir + "result_csv/mae_comparison.csv", index=True)


if __name__ == '__main__':
    run_bidmc()
    run_capno()
    run_vitalconnect_s3()
