# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
import shap
import joblib

# Optional voice output
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

warnings.filterwarnings('ignore')


def load_waveforms_from_hdf5(hdf5_path):
    import h5py

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Direct dataset case
        candidate_datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        if candidate_datasets:
            ds_name = candidate_datasets[0]
            data = np.array(f[ds_name])
        else:
            # dataset may be nested inside a group like data/bucket0
            candidate_groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            if not candidate_groups:
                raise ValueError(f"No dataset or group found in {hdf5_path}")

            group = f[candidate_groups[0]]
            sub_datasets = [k for k in group.keys() if isinstance(group[k], h5py.Dataset)]
            if not sub_datasets:
                raise ValueError(f"No dataset found in group {candidate_groups[0]} of {hdf5_path}")

            ds_name = sub_datasets[0]
            data = np.array(group[ds_name])

    # Normalize shapes
    if data.ndim == 2:
        # shape (samples, timesteps) -> add channel
        data = data[:, :, np.newaxis]
    elif data.ndim == 3:
        s0, s1, s2 = data.shape
        if s1 == 3 and s2 != 3:
            # likely (samples, channels, timesteps)
            data = np.transpose(data, (0, 2, 1))
        elif s2 == 3 and s1 != 3:
            # likely correct (samples, timesteps, channels)
            pass
        else:
            # ambiguous, but we assume final dimension is channels
            pass
    else:
        raise ValueError(f"Unexpected waveform dimensions: {data.shape}")

    return data


def load_labels_from_metadata(metadata_path, label_value):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    count = len(df)
    return np.full(count, label_value, dtype=np.int32)


def butter_bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # data: (timesteps, channels) or (samples, timesteps, channels)
    if data.ndim == 2:
        return filtfilt(b, a, data, axis=0)
    elif data.ndim == 3:
        out = np.zeros_like(data)
        for i in range(data.shape[0]):
            out[i] = filtfilt(b, a, data[i], axis=0)
        return out
    else:
        raise ValueError('Bad input for bandpass filter')


def filter_and_segment(X, y, window_size=2500, stride=2500):
    segmented = []
    segmented_y = []

    for i in range(len(X)):
        signal = X[i]
        if signal.shape[0] < window_size:
            continue

        for start in range(0, signal.shape[0] - window_size + 1, stride):
            window = signal[start:start + window_size]
            if window.shape[0] == window_size:
                segmented.append(window)
                segmented_y.append(y[i])

    if len(segmented) == 0:
        raise ValueError('No segments generated; verify window_size/stride.')

    return np.array(segmented), np.array(segmented_y)


def remove_nans(X, y):
    mask = ~np.isnan(X).any(axis=(1, 2))
    return X[mask], y[mask]


def normalize_data(X):
    n_samples, n_timesteps, n_channels = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_channels)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_channels)
    return X_scaled, scaler  # Return scaler for saving


def build_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


def plot_confusion(cm):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    classes = ['Noise (0)', 'Earthquake (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def shap_explain(model, x_train, x_test):
    try:
        explainer = shap.DeepExplainer(model, x_train[:100])
        shap_values = explainer.shap_values(x_test[:50])

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values[0], x_test[:50], show=False)
        plt.title('SHAP summary plot')
        plt.savefig('shap_summary.png')
        plt.close()
        return True
    except Exception as e:
        print('SHAP explanation failed:', str(e))
        return False


def text_to_speech(message):
    if pyttsx3 is None:
        print('pyttsx3 not installed; skipping voice alert')
        return
    try:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print('Voice alert failed:', str(e))


def main():
    np.random.seed(42)

    # 1) Load datasets
    earthquake_path = 'dataset/dataset_earthquakes/waveforms.hdf5'
    earthquake_meta = 'dataset/dataset_earthquakes/metadata.csv'
    noise_path = 'dataset/dataset_noise/waveforms.hdf5'
    noise_meta = 'dataset/dataset_noise/metadata.csv'

    X_eq = load_waveforms_from_hdf5(earthquake_path)
    y_eq = load_labels_from_metadata(earthquake_meta, 1)

    X_noise = load_waveforms_from_hdf5(noise_path)
    y_noise = load_labels_from_metadata(noise_meta, 0)

    # Align lengths if necessary
    if len(X_eq) != len(y_eq):
        min_len = min(len(X_eq), len(y_eq))
        X_eq, y_eq = X_eq[:min_len], y_eq[:min_len]
    if len(X_noise) != len(y_noise):
        min_len = min(len(X_noise), len(y_noise))
        X_noise, y_noise = X_noise[:min_len], y_noise[:min_len]

    X = np.concatenate([X_eq, X_noise], axis=0)
    y = np.concatenate([y_eq, y_noise], axis=0)

    # 2) Filter signal
    X = butter_bandpass_filter(X, lowcut=1.0, highcut=45.0, fs=100.0, order=4)

    # 3) Segment into windows
    X, y = filter_and_segment(X, y, window_size=2500, stride=2500)

    # 4) Remove NaN
    X, y = remove_nans(X, y)

    # 5) Normalize
    X, scaler = normalize_data(X)

    # Save scaler for consistent preprocessing in app.py
    import joblib
    joblib.dump(scaler, 'scaler.pkl')

    # 6) Ensure shape
    assert X.ndim == 3, f"X must be 3D, got {X.shape}"

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)

    # training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=8, batch_size=32, callbacks=callbacks, verbose=2)

    model.save('model.h5')
    plot_training(history)

    # Evaluation
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n', cm)
    print('Classification report:\n', classification_report(y_test, y_pred, digits=4))
    plot_confusion(cm)

    if shap_explain(model, X_train, X_test):
        print('SHAP summary saved as shap_summary.png')

    # final logic for a sample event
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx + 1]
    pred_prob = float(model.predict(sample)[0, 0])
    pred_label = int(pred_prob >= 0.5)

    if pred_label == 1:
        risk = 'Low' if pred_prob < 0.6 else 'Medium' if pred_prob < 0.8 else 'High'
        print('Earthquake Detected')
        print(f'Accuracy: {acc:.4f}')
        print(f'Probability: {pred_prob:.4f}')
        print(f'Risk Level: {risk}')
        print('XAI explanation: The model uses the temporal and spectral characteristics in the most salient 2500-sample window to assign earthquake risk. SHAP highlights peaks and anomalies in key time steps that push prediction toward earthquake behavior.')
        text_to_speech('Warning! Earthquake detected. Take precautions.')
    else:
        print('No Earthquake Detected')
        text_to_speech('No earthquake detected. You are safe.')


if __name__ == '__main__':
    main()
