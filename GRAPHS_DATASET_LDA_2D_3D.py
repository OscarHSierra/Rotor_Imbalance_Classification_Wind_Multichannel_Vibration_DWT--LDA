import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# -----------------------------
# LOAD DATA
# -----------------------------
dataset = np.load("datasetmarzo.npy")
labels = np.load("labelsmarzo.npy")

dataset = dataset.reshape(dataset.shape[0], 3200, 24)


# -----------------------------
# DWT FEATURES (BEST CONFIG)
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


print("Extracting features...")
X = dwt_features(dataset)


# -----------------------------
# NORMALIZATION
# -----------------------------
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X = (X - mean) / std


# -----------------------------
# STYLE (IEEE-LIKE)
# -----------------------------
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})


# =============================
# LDA 2D (PUBLICATION READY)
# =============================
lda_2d = LDA(n_components=2)
X_2d = lda_2d.fit_transform(X, labels)

plt.figure(figsize=(6,5))

palette = sns.color_palette("colorblind", len(np.unique(labels)))

sns.scatterplot(
    x=X_2d[:,0],
    y=X_2d[:,1],
    hue=labels,
    palette=palette,
    s=35,
    edgecolor='k',
    linewidth=0.3
)

plt.title("LDA Projection (2D)")
plt.xlabel("LD1")
plt.ylabel("LD2")

plt.legend(title="Class", loc="best", frameon=True)

plt.tight_layout()

plt.savefig("LDA_2D.png", dpi=300, bbox_inches='tight')

plt.show()


from mpl_toolkits.mplot3d import Axes3D

lda_3d = LDA(n_components=3)
X_3d = lda_3d.fit_transform(X, labels)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_3d[:,0],
    X_3d[:,1],
    X_3d[:,2],
    c=labels,
    cmap="tab10",
    s=30,
    edgecolor='k'
)

ax.set_title("LDA Projection (3D)")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_zlabel("LD3")

plt.tight_layout()

plt.savefig("LDA_3D.png", dpi=300, bbox_inches='tight')

plt.show()