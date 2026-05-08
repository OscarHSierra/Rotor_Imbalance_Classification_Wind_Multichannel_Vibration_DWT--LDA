import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -----------------------------
# LOAD DATA
# -----------------------------
dataset = np.load("datasetmarzo.npy")
labels = np.load("labelsmarzo.npy")
groups = np.load("groupsmarzo.npy")

dataset = dataset.reshape(dataset.shape[0], 3200, 24)


# -----------------------------
# DWT FEATURES 
# -----------------------------
def dwt_features(data, wavelet="sym5", level=4):
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


print("Extracting DWT features (sym5, level 4)...")
X = dwt_features(dataset)


# -----------------------------
# CONFIGURATION
# -----------------------------
lda_components = [1, 2, 3, 4]

gkf = GroupKFold(n_splits=5)

results = []


# -----------------------------
# EXPERIMENT LOOP
# -----------------------------
for n_comp in lda_components:

    print(f"\nLDA components = {n_comp}")

    acc_list = []

    for train_idx, test_idx in gkf.split(X, labels, groups):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # -----------------------------
        # NORMALIZATION (train only)
        # -----------------------------
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # -----------------------------
        # LDA
        # -----------------------------
        lda = LDA(n_components=n_comp)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)

        # -----------------------------
        # SVM
        # -----------------------------
        model = SVC(kernel='rbf', C=10, gamma='scale')
        model.fit(X_train_lda, y_train)

        y_pred = model.predict(X_test_lda)

        acc_list.append(accuracy_score(y_test, y_pred))

    results.append({
        "LDA_components": n_comp,
        "Mean_Accuracy": np.mean(acc_list),
        "Std_Accuracy": np.std(acc_list)
    })


# -----------------------------
# RESULTS
# -----------------------------
df = pd.DataFrame(results)
print(df)


# -----------------------------
# PLOT (CLEAN VERSION)
# -----------------------------
plt.figure(figsize=(8,5))

plt.errorbar(
    df["LDA_components"],
    df["Mean_Accuracy"],
    yerr=df["Std_Accuracy"],
    marker='o',
    linewidth=2,
    capsize=5,
    label="Mean Accuracy ± Std"
)

plt.xlabel("Number of LDA Components")
plt.ylabel("Accuracy")
plt.title("Effect of LDA Dimensionality (DWT sym5 level 4, SVM)")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()