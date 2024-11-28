import torch

import os

# Ruta de la carpeta 'data'
data_folder = './data'

# Comprovar si la carpeta existeix
if os.path.exists(data_folder):
    # Llistar els fitxers dins de la carpeta
    files = os.listdir(data_folder)
    print("Fitxers dins de la carpeta 'data':")
    for file in files:
        print(f"- {file}")
else:
    print(f"La carpeta {data_folder} no existeix.")
