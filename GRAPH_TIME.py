import numpy as np
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt

archivos = sorted(glob.glob(
'/Users/dspsogamoso/Library/CloudStorage/OneDrive-SharedLibraries-ONEDRIVE/Doctorado/DATASET TURBINA/ventanas/WAVELET/REVISADO_MARZO_17/dataset15/*.mat'
))

window = 3200

senal_sana = None
senal_falla = None


for idx, archivo in enumerate(archivos):

    data = loadmat(archivo)

    # usar un solo sensor (A1X)
    signal = data['A1X'].flatten()
    signal = signal - np.mean(signal)

    # etiquetas
    if idx < 15:
        label = 0
    elif idx < 20:
        label = 1
    elif idx < 25:
        label = 2
    elif idx < 30:
        label = 3
    else:
        label = 4

    # guardar ejemplos
    if label == 0 and senal_sana is None:
        senal_sana = signal[:window]

    if label == 1 and senal_falla is None:
        senal_falla = signal[:window]

    # detener cuando ya tenga ambas
    if senal_sana is not None and senal_falla is not None:
        break


plt.figure(figsize=(12,5))

plt.plot(senal_sana, label="Healthy Condition", linewidth=1)
plt.plot(senal_falla, label="Fault Condition", linewidth=1)

plt.title("Raw Vibration Signals (Time Domain)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()