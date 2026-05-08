#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pywt
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# CONFIG
# =========================
N_BANDS = 10
BANDS_LIST = [5, 10, 15, 20]

# =========================
# LOAD
# =========================
X = np.load("datasetabril.npy")
y = np.load("labelsabril.npy")
groups = np.load("groupsabril.npy")

X = X.reshape(X.shape[0], 3200, 24)

gkf = GroupKFold(n_splits=5)

# =========================
# FEATURES
# =========================
def dwt_features(X):
    features = []
    for ventana in X:
        feat = []
        for ch in range(ventana.shape[1]):
            señal = ventana[:, ch]
            coeffs = pywt.wavedec(señal, 'sym5', level=4)
            cA4, cD4, cD3, cD2, cD1 = coeffs
            for c in [cA4, cD4, cD3, cD2, cD1]:
                feat.append(np.sum(c**2))
        features.append(feat)
    return np.array(features)

def fft_features(X, n_bands=N_BANDS):
    features = []
    for ventana in X:
        feat = []
        for ch in range(ventana.shape[1]):
            señal = ventana[:, ch]
            fft_vals = np.fft.rfft(señal)
            fft_mag = np.abs(fft_vals)**2
            bandas = np.array_split(fft_mag, n_bands)
            for b in bandas:
                feat.append(np.sum(b))
        features.append(feat)
    return np.array(features)

# =========================
# EVALUATION
# =========================
def evaluar(nombre="DWT", n_bands=N_BANDS):

    accs, tiempos = [], []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()
        if nombre == "DWT":
            X_train = dwt_features(X_train_raw)
            X_test = dwt_features(X_test_raw)
        else:
            X_train = fft_features(X_train_raw, n_bands)
            X_test = fft_features(X_test_raw, n_bands)
        tiempos.append(time.time() - start)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return {
        "accs": accs,
        "acc_mean": np.mean(accs),
        "time": np.mean(tiempos),
        "y_true": y_true_all,
        "y_pred": y_pred_all
    }

# =========================
# RESULTS
# =========================
res_dwt = evaluar("DWT")
res_fft = evaluar("FFT", n_bands=10)

# =========================
# 1. CONFUSION MATRICES
# =========================
cm_dwt = confusion_matrix(res_dwt["y_true"], res_dwt["y_pred"])

plt.figure(figsize=(5,4))
ConfusionMatrixDisplay(cm_dwt).plot()
plt.title("Confusion Matrix - DWT")
plt.savefig("cm_dwt.png", dpi=300)
plt.show()

cm_fft = confusion_matrix(res_fft["y_true"], res_fft["y_pred"])

plt.figure(figsize=(5,4))
ConfusionMatrixDisplay(cm_fft).plot()
plt.title("Confusion Matrix - FFT (10 bands)")
plt.savefig("cm_fft.png", dpi=300)
plt.show()

# =========================
# 2. BOXPLOT
# =========================
plt.figure(figsize=(6,4))
plt.boxplot([res_dwt["accs"], res_fft["accs"]])
plt.xticks([1,2], ["DWT", "FFT"])
plt.ylabel("Accuracy")
plt.title("Model Stability Across Folds")
plt.grid(True)

plt.savefig("boxplot_accuracy.png", dpi=300)
plt.show()

# =========================
# 3. TRADE-OFF
# =========================
acc_vs_bands = []
time_vs_bands = []

for b in BANDS_LIST:
    res = evaluar("FFT", n_bands=b)
    acc_vs_bands.append(res["acc_mean"])
    time_vs_bands.append(res["time"])

fig, ax1 = plt.subplots(figsize=(6,4))

# Accuracy
color_acc = 'tab:blue'
ax1.set_xlabel("Number of Bands")
ax1.set_ylabel("Accuracy", color=color_acc)
ax1.plot(BANDS_LIST, acc_vs_bands, marker='o', color=color_acc)
ax1.tick_params(axis='y', labelcolor=color_acc)

# Time
ax2 = ax1.twinx()
color_time = 'tab:red'
ax2.set_ylabel("Time (s)", color=color_time)
ax2.plot(BANDS_LIST, time_vs_bands, linestyle='--', marker='s', color=color_time)
ax2.tick_params(axis='y', labelcolor=color_time)

plt.title("Accuracy vs Computational Cost Trade-off")
plt.grid(True)

plt.savefig("tradeoff_fft.png", dpi=300)
plt.show()

# =========================
# 4. PCA
# =========================
def plot_pca(X_feat, y, title, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_feat)

    plt.figure(figsize=(6,4))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", s=10)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, dpi=300)
    plt.show()

# =========================
# 5. t-SNE
# =========================
def plot_tsne(X_feat, y, title, filename):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_feat)

    plt.figure(figsize=(6,4))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Class {label}", s=10)

    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, dpi=300)
    plt.show()

# =========================
# Complete Features
# =========================
X_dwt_full = dwt_features(X)
X_fft_full = fft_features(X, n_bands=10)

# PCA
plot_pca(X_dwt_full, y, "PCA - DWT", "pca_dwt.png")
plot_pca(X_fft_full, y, "PCA - FFT (10 bands)", "pca_fft.png")

# t-SNE
plot_tsne(X_dwt_full, y, "t-SNE - DWT", "tsne_dwt.png")
plot_tsne(X_fft_full, y, "t-SNE - FFT (10 bands)", "tsne_fft.png")