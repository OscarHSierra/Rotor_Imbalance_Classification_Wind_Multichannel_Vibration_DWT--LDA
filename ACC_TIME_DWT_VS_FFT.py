import numpy as np
import pywt
import time
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# =========================
# CONFIGURATION
# =========================
N_BANDS = 1››0   

# =========================
# DATA LOAD
# =========================
X = np.load("datasetabril.npy")
y = np.load("labelsabril.npy")
groups = np.load("groupsabril.npy")

# reshape (ventanas, 3200, 24)
X = X.reshape(X.shape[0], 3200, 24)

print("Shape:", X.shape)

# =========================
# FEATURES DWT 
# =========================
def dwt_features(X):
    features = []

    for ventana in X:
        feat_ventana = []

        for ch in range(ventana.shape[1]):
            señal = ventana[:, ch]

            coeffs = pywt.wavedec(señal, 'sym5', level=4)

            # explícito: [cA4, cD4, cD3, cD2, cD1]
            cA4, cD4, cD3, cD2, cD1 = coeffs
            subbands = [cA4, cD4, cD3, cD2, cD1]

            for c in subbands:
                energia = np.sum(c**2)
                feat_ventana.append(energia)

        features.append(feat_ventana)

    return np.array(features)

# =========================
# FEATURES FFT
# =========================
def fft_features(X, n_bands=N_BANDS):
    features = []

    for ventana in X:
        feat_ventana = []

        for ch in range(ventana.shape[1]):
            señal = ventana[:, ch]

            fft_vals = np.fft.rfft(señal)
            fft_mag = np.abs(fft_vals)**2

           
            bandas = np.array_split(fft_mag, n_bands)

            for banda in bandas:
                energia = np.sum(banda)
                feat_ventana.append(energia)

        features.append(feat_ventana)

    return np.array(features)

# =========================
# VALIDATION
# =========================
gkf = GroupKFold(n_splits=5)

# =========================
# EVALUATION
# =========================
def evaluar_metodo(nombre="DWT"):

    accs = []
    f1s = []
    tiempos = []

    for train_idx, test_idx in gkf.split(X, y, groups):

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

     
        start = time.time()

        if nombre == "DWT":
            X_train = dwt_features(X_train_raw)
        else:
            X_train = fft_features(X_train_raw)

        train_time = time.time() - start

        start = time.time()

        if nombre == "DWT":
            X_test = dwt_features(X_test_raw)
        else:
            X_test = fft_features(X_test_raw)

        test_time = time.time() - start

        tiempos.append(train_time + test_time)

        #NORMALIZATION
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # MODEL
        model = SVC(kernel='rbf')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    print(f"\n--- {nombre} ---")
    print("Accuracy:", np.mean(accs), "±", np.std(accs))
    print("F1-score:", np.mean(f1s), "±", np.std(f1s))
    print("Tiempo promedio:", np.mean(tiempos))
    print("Features:", X_train.shape[1])

# EEXECUTION
evaluar_metodo("DWT")
evaluar_metodo("FFT")

# TEST 
print("\n=== TEST SANIDAD (labels aleatorios) ===")

y_random = np.random.permutation(y)
accs = []

for train_idx, test_idx in gkf.split(X, y_random, groups):

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    y_train = y_random[train_idx]
    y_test = y_random[test_idx]

    X_train = dwt_features(X_train_raw)
    X_test = dwt_features(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accs.append(accuracy_score(y_test, y_pred))

print("Accuracy random:", np.mean(accs), "±", np.std(accs))

# INFO
print("\n INFO FEATURES ")
print("DWT features:", dwt_features(X[:1]).shape[1])
print("FFT features:", fft_features(X[:1]).shape[1])