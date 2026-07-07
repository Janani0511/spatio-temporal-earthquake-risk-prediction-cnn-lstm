import h5py, pandas as pd
for d in ['dataset/dataset_earthquakes','dataset/dataset_noise']:
    print('---', d)
    with h5py.File(d + '/waveforms.hdf5', 'r') as f:
        print('keys', list(f.keys()))
        for k in f.keys():
            print(k, f[k].shape, f[k].dtype)
    print('metadata rows', pd.read_csv(d + '/metadata.csv').shape)
