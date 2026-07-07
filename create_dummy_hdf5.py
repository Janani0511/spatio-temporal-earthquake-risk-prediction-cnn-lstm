import h5py
import numpy as np

# Create dummy data for earthquakes
num_samples = 100  # small number for testing
shape = (num_samples, 3, 25001)
data = np.random.randn(*shape).astype(np.float32)

with h5py.File('dataset/dataset_earthquakes/waveforms.hdf5', 'w') as f:
    f.create_dataset('bucket0', data=data)

# Create dummy data for noise
data_noise = np.random.randn(*shape).astype(np.float32)

with h5py.File('dataset/dataset_noise/waveforms.hdf5', 'w') as f:
    f.create_dataset('bucket0', data=data_noise)

print("Dummy HDF5 files created.")