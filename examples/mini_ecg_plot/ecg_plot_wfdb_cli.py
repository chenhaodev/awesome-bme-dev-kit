import wfdb
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_ekg(record_name, channel_id, start_sec, end_sec):
    """
    Plot EKG signal for a given record and time range.

    :param record_name: The name of the record to plot
    :param channel_id: The channel of the record to plot
    :param start_sec: The start time in seconds for the plot
    :param end_sec: The end time in seconds for the plot
    """
    # Read the header information
    header = wfdb.rdheader(record_name)
    fs = header.fs

    # Read the EKG data
    signals, _ = wfdb.rdsamp(record_name, channels=[channel_id], sampfrom=int(start_sec * fs), sampto=int(end_sec * fs))

    # Calculate the timestamps for each sample
    num_samples = signals.shape[0]
    timestamps = np.arange(start_sec, end_sec, 1/fs)[:num_samples]

    # Plotting the EKG signal within the specified time range
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, signals)
    plt.xlabel('Time (seconds)')
    plt.ylabel('EKG Signal')
    plt.title(f'EKG Signal from {record_name} between {start_sec} and {end_sec} seconds')
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to parse command-line arguments and call the plot_ekg function.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot EKG signal for a given record and time range.')
    parser.add_argument('-f', '--record_name', type=str, help='The filename of the record to plot')
    parser.add_argument('-c', '--channel', type=int, default=0, help='The channel of the record to plot')
    parser.add_argument('-on', '--onset', type=float, required=True, help='The start time in seconds for the plot')
    parser.add_argument('-off', '--offset', type=float, required=True, help='The end time in seconds for the plot')

    # Parse arguments
    args = parser.parse_args()

    # Call the plot function with the provided arguments
    plot_ekg(args.record_name, args.channel, args.onset, args.offset)

if __name__ == '__main__':
    main()
