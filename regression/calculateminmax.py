import os
import numpy as np
from tqdm import tqdm  # Importa tqdm per la barra di progresso

# Funzione per calcolare i minimi e massimi dei file .npy
def calculate_min_max(dataset_path):
    # Definire i minimi e massimi iniziali
    min_exp, max_exp = float('inf'), float('-inf')
    min_globalpose, max_globalpose = float('inf'), float('-inf')
    min_jawpose, max_jawpose = float('inf'), float('-inf')

    # Scansiona tutte le cartelle nel dataset
    all_subjects = os.listdir(dataset_path)
    for subject in tqdm(all_subjects, desc="Elaborazione soggetti"):  # Barra di progresso per i soggetti
        subject_path = os.path.join(dataset_path, subject)
        if os.path.isdir(subject_path):
            for action_unit in os.listdir(subject_path):
                action_unit_path = os.path.join(subject_path, action_unit)
                if os.path.isdir(action_unit_path):
                    for video in os.listdir(action_unit_path):
                        video_path = os.path.join(action_unit_path, video)
                        if os.path.isdir(video_path):
                            for frame_folder in os.listdir(video_path):
                                frame_path = os.path.join(video_path, frame_folder)
                                if os.path.isdir(frame_path):
                                    # Carica i file .npy (exp.npy, globalpose.npy, jawpose.npy)
                                    try:
                                        exp_path = os.path.join(frame_path, 'exp.npy')
                                        globalpose_path = os.path.join(frame_path, 'globalpose.npy')
                                        jawpose_path = os.path.join(frame_path, 'jawpose.npy')

                                        # Controlla se i file esistono
                                        if os.path.exists(exp_path) and os.path.exists(globalpose_path) and os.path.exists(jawpose_path):
                                            exp = np.load(exp_path)
                                            globalpose = np.load(globalpose_path)
                                            jawpose = np.load(jawpose_path)

                                            # Aggiorna i minimi e massimi per exp.npy
                                            min_exp = min(min_exp, exp.min())
                                            max_exp = max(max_exp, exp.max())

                                            # Aggiorna i minimi e massimi per globalpose.npy
                                            min_globalpose = min(min_globalpose, globalpose.min())
                                            max_globalpose = max(max_globalpose, globalpose.max())

                                            # Aggiorna i minimi e massimi per jawpose.npy
                                            min_jawpose = min(min_jawpose, jawpose.min())
                                            max_jawpose = max(max_jawpose, jawpose.max())

                                    except Exception as e:
                                        print(f"Errore nel caricare i file per il frame: {frame_path}, errore: {e}")

    return (min_exp, max_exp), (min_globalpose, max_globalpose), (min_jawpose, max_jawpose)

# Percorso del dataset
dataset_path = "/andromeda/personal/fgastaldi/facemorphic_output"

# Calcola i minimi e massimi
exp_range, globalpose_range, jawpose_range = calculate_min_max(dataset_path)

# Stampa i risultati
print(f"Range per exp.npy: Min = {exp_range[0]}, Max = {exp_range[1]}")
print(f"Range per globalpose.npy: Min = {globalpose_range[0]}, Max = {globalpose_range[1]}")
print(f"Range per jawpose.npy: Min = {jawpose_range[0]}, Max = {jawpose_range[1]}")
