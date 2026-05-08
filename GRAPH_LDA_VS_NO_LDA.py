import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset = np.load("datasetmarzo.npy")
labels = np.load("labelsmarzo.npy")
groups = np.load("groupsmarzo.npy")

dataset = dataset.reshape(dataset.shape[0], 3200, 24)

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


wavelet = "sym5"   # 
levels = [1, 2, 3, 4]
methods = ["WPT", "DWT"]

gkf = GroupKFold(n_splits=5)

results = []

for method in methods:
    for level in levels:

        print(f"\n{method} - {wavelet} - level {level}")

        # features
        if method == "WPT":
            X = wpt_features(dataset, wavelet, level)
        else:
            X = dwt_features(dataset, wavelet, level)

        acc_no_lda = []
        acc_lda = []

        for train_idx, test_idx in gkf.split(X, labels, groups):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8

            X_train_n = (X_train - mean) / std
            X_test_n = (X_test - mean) / std

            # -----------------------------
            # NO LDA
            # -----------------------------
            model = SVC(kernel='rbf', C=10, gamma='scale')
            model.fit(X_train_n, y_train)

            y_pred = model.predict(X_test_n)
            acc_no_lda.append(accuracy_score(y_test, y_pred))

            # -----------------------------
            # LDA
            # -----------------------------
            lda = LDA(n_components=4)
            X_train_lda = lda.fit_transform(X_train_n, y_train)
            X_test_lda = lda.transform(X_test_n)

            model.fit(X_train_lda, y_train)
            y_pred = model.predict(X_test_lda)

            acc_lda.append(accuracy_score(y_test, y_pred))

        results.append({
            "Method": method,
            "Level": level,
            "No_LDA": np.mean(acc_no_lda),
            "With_LDA": np.mean(acc_lda)
        })


# -----------------------------
# RESULTS
# -----------------------------
df = pd.DataFrame(results)
print(df)

plt.figure(figsize=(10,6))

for method in methods:

    sub = df[df["Method"] == method]

    # NO LDA
    plt.plot(
        sub["Level"], sub["No_LDA"],
        marker='o', linestyle='--',
        linewidth=2,
        label=f"{method} - NO LDA"
    )

    # LDA
    plt.plot(
        sub["Level"], sub["With_LDA"],
        marker='o', linestyle='-',
        linewidth=2,
        label=f"{method} - WITH LDA"
    )


plt.xlabel("Wavelet Level")
plt.ylabel("Accuracy")
plt.title("LDA impact (wavelet = sym5, SVM)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()