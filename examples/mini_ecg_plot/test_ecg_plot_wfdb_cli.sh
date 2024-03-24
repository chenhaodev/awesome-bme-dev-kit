wget -r -np https://physionet.org/files/mitdb/1.0.0/119.hea; wget -r -np https://physionet.org/files/mitdb/1.0.0/119.dat
python ecg_plot_wfdb_cli.py -f physionet.org/files/mitdb/1.0.0/119 -on 0.0 -off 10.0
