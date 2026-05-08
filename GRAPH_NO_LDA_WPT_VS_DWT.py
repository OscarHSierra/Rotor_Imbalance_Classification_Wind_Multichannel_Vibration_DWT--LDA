import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -----------------------------
# DATA LOAD
# -----------------------------
dataset = np.load("datasetmarzo.npy")
labels = np.load("labelsmarzo.npy")
groups = np.load("groupsmarzo.npy")

dataset = dataset.reshape(dataset.shape[0], 3200, 24)


# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------

def wpt_features(data, wavelet, level):
    features = []

    for window in data:
        feat_window = []

        for ch in range(window.shape[1]):
            signal = window[:, ch]

            wp = pywt.WaveletPacket(
                data=signal,
                wavelet=wavelet,
                mode="symmetric",
                maxlevel=level
            )

            nodes = wp.get_level(level, order="freq")

            for node in nodes:
                feat_window.append(np.sum(node.data**2))

        features.append(feat_window)

    return np.array(features)


def dwt_features(data, wavelet, level):
    features = []

    for window in data:
        feat_window = []

        for ch in range(window.shape[1]):
            signal = window[:, ch]

            coeffs = pywt.wavedec(signal, wavelet, level=level)

            for c in coeffs:
                feat_window.append(np.sum(c**2))

        features.append(feat_window)

    return np.array(features)


# -----------------------------
# CONFIGURATION
# -----------------------------
wavelets = ["haar", "db4", "sym5"]
levels = [1, 2, 3, 4]
methods = ["WPT", "DWT"]

gkf = GroupKFold(n_splits=5)

results = []


# -----------------------------
# EXPERIMENTAL LOOP
# -----------------------------
for method in methods:
    for wavelet in wavelets:
        for level in levels:

            print(f"\n{method} - {wavelet} - level {level}")

            # features
            if method == "WPT":
                X = wpt_features(dataset, wavelet, level)
            else:
                X = dwt_features(dataset, wavelet, level)

            acc_list = []

            for train_idx, test_idx in gkf.split(X, labels, groups):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # -----------------------------
                # NORMALIZACIÓN (igual que antes)
                # -----------------------------
                mean = X_train.mean(axis=0)
                std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - mean) / std
                X_test = (X_test - mean) / std

                # -----------------------------
                # SVM (SIN LDA)
                # -----------------------------
                model = SVC(kernel='rbf', C=10, gamma='scale')
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc_list.append(accuracy_score(y_test, y_pred))

            results.append({
                "Method": method,
                "Wavelet": wavelet,
                "Level": level,
                "Accuracy": np.mean(acc_list)
            })


# -----------------------------
# RESULTS
# -----------------------------
df = pd.DataFrame(results)
print(df)


# -----------------------------
# FINAL GRAPH
# -----------------------------
plt.figure(figsize=(12,6))

for method in methods:
    for wavelet in wavelets:

        subset = df[(df["Method"] == method) & (df["Wavelet"] == wavelet)]

        plt.plot(
            subset["Level"],
            subset["Accuracy"],
            marker='o',
            linewidth=2,
            label=f"{method}-{wavelet}"
        )

plt.xlabel("Level")
plt.ylabel("Accuracy")
plt.title("WPT vs DWT (SVM) - SIN LDA")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()