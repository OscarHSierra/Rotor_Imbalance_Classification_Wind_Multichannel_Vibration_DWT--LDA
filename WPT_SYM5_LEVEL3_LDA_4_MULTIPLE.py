import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# modelos
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# -----------------------------
# DATA LOAD
# -----------------------------
dataset = np.load("datasetmarzo.npy")   # sin normalizar
labels = np.load("labelsmarzo.npy")
# =============================================================================
groups = np.load("groupsmarzo.npy")
# =============================================================================

print("Dataset:", dataset.shape)

# reconstruir a (N,3200,24)
dataset = dataset.reshape(dataset.shape[0], 3200, 24)


# -----------------------------
# WPT FEATURES
# -----------------------------
def wavelet_energy_features(data, wavelet="sym5", level=3):

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


print("Extrayendo WPT...")
X = wavelet_energy_features(dataset)

print("Features:", X.shape)


# -----------------------------
# MODELS
# -----------------------------
models = {
    "SVM": SVC(kernel='rbf', C=10, gamma='scale'),
    "Random Forest": RandomForestClassifier(n_estimators=150),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}


# -----------------------------
# GROUP K-FOLD
# -----------------------------
gkf = GroupKFold(n_splits=5)

results = []
n_classes = len(np.unique(labels))


for model_name, model in models.items():


    print(f"Modelo: {model_name}")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    cm_total = np.zeros((n_classes, n_classes))

    fold = 1

    for train_idx, test_idx in gkf.split(X, labels, groups):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # -----------------------------
        # NORMALIZATION
        # -----------------------------
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # -----------------------------
        # LDA 
        # -----------------------------
        lda = LDA(n_components=4)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)

        # -----------------------------
        # TRAINING
        # -----------------------------
        model.fit(X_train_lda, y_train)
        y_pred = model.predict(X_test_lda)

        # metrics
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, average='weighted'))
        rec_list.append(recall_score(y_test, y_pred, average='weighted'))
        f1_list.append(f1_score(y_test, y_pred, average='weighted'))

        # -----------------------------
        # Confusion matrix
        # -----------------------------
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
        cm_total += cm

        # -----------------------------
        # matrix per fold
        # -----------------------------
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(f"{model_name} - Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()
        plt.show()

        fold += 1

    # -----------------------------
    # Total matriz- without normalization
    # -----------------------------
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_total, annot=True, fmt=".0f", cmap="Blues")

    plt.title(f"{model_name} - Confusion Matrix Total (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()

    # guardar resultados
    results.append({
        "Model": model_name,
        "Accuracy": np.mean(acc_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1 Score": np.mean(f1_list)
    })


# -----------------------------
# RESULTS
# -----------------------------
df_results = pd.DataFrame(results)

print("\nResultados finales:")
print(df_results)


# -----------------------------
# GRAPH
# -----------------------------
df_melt = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10,6))

sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric")

plt.title("Comparación de Clasificadores (WPT sym5 + LDA + GroupKFold)")
plt.ylim(0, 1)

plt.xticks(rotation=15)
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()