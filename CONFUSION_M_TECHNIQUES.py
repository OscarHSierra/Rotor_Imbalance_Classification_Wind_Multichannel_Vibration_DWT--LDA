import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# modelos
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# -----------------------------
# LOAD DATA
# -----------------------------
dataset = np.load("datasetmarzo.npy")
labels = np.load("labelsmarzo.npy")
groups = np.load("groupsmarzo.npy")

dataset = dataset.reshape(dataset.shape[0], 3200, 24)


# -----------------------------
# DWT FEATURES-sym5-level 4
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
# MODELS
# -----------------------------
models = {
    "SVM": SVC(kernel='rbf', C=10, gamma='scale'),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}


# -----------------------------
# CROSS VALIDATION
# -----------------------------
gkf = GroupKFold(n_splits=5)
n_classes = len(np.unique(labels))

results = []


for model_name, model in models.items():

    print(f"\n====================")
    print(f"Model: {model_name}")
    print(f"====================")

    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    cm_total = np.zeros((n_classes, n_classes))

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
        # LDA (best = 3 components)
        # -----------------------------
        lda = LDA(n_components=4)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # metrics
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # confusion matrix (fixed size)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
        cm_total += cm

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    results.append({
        "Model": model_name,
        "Accuracy": np.mean(acc_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1 Score": np.mean(f1_list)
    })

    # -----------------------------
    # CONFUSION MATRIX (TOTAL COUNTS)
    # -----------------------------
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_total, annot=True, fmt=".0f", cmap="Blues")

    plt.title(f"{model_name} - Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()


# -----------------------------
# RESULTS TABLE
# -----------------------------
df_results = pd.DataFrame(results)

print("\nFinal Results:")
print(df_results)


# -----------------------------
# BAR PLOT (METRICS)
# -----------------------------
df_melt = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10,6))

sns.barplot(
    data=df_melt,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Classifier Comparison (DWT + LDA + GroupKFold)")
plt.ylim(0, 1)

plt.xticks(rotation=15)
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()