import os
import argparse
from pathlib import Path
from tqdm import tqdm, auto

import inferno
from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
from inferno_apps.FaceReconstruction.utils.output import save_obj, save_images, save_codes
from inferno_apps.FaceReconstruction.utils.test import test
from inferno.utils.other import get_path_to_assets


def process_folder(input_folder, output_folder, model, conf, args):
    try:
        dataset = TestData(str(input_folder), face_detector = "fan", max_detection = 20) # Permette di creare un dataset a partire dalle immagini contenute nella cartella "input_folder"
        # Il parametro "max_detection = 20" indica il numero massimo di volti rilevabili per immagine;
        # "fan" è il tipo di detector utilizzato per rilevare i volti nelle immagini
    except Exception as e:
        print(f"Error while creating dataset from {input_folder}: {e}. Skipping folder.")
        return
    
    for i in auto.tqdm(range(len(dataset))):
        try:
            batch = dataset[i]
            vals = test(model, batch)
            visdict = model.visualize_batch(batch, i, None, in_batch_idx = None)
            current_bs = batch["image"].shape[0]

            for j in range(current_bs):
                try:
                    name = batch["image_name"][j]
                    sample_output_folder = Path(output_folder) / name
                    sample_output_folder.mkdir(parents = True, exist_ok = True)

                    if args.save_mesh:
                        save_obj(model, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
                    if args.save_codes:
                        save_codes(Path(output_folder), name, vals, i = j)
                    if args.save_images:
                        save_images(output_folder, name, visdict, with_detection = True, i = j)
                except Exception as e:
                    print(f"Error saving frame {name}: {e}. Skipping frame.")
        except Exception as e:
            print(f"Error processing frame {i} in {input_folder}: {e}. Skipping frame.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type = str, default = str("/home/fgastaldi/facemorphic_annotated_rgb/recordings")) # Viene richiesto il percorso della cartella da qui prelevare le immagini da analizzare
    parser.add_argument('--output_folder', type = str, default = "/andromeda/personal/fgastaldi/facemorphic_output", help = "Output folder to save the results to.") # Cartella in cui salvare i risultati
    parser.add_argument('--model_name', type = str, default = 'EMICA-CVT_flame2020_notexture', help = 'Name of the model to use.') # Nome del modello di ricostruzione facciale da utilizzare
    parser.add_argument('--path_to_models', type = str, default = str(Path(get_path_to_assets()) / "FaceReconstruction/models")) # Percorso che riporta a tutti i modelli salvati (che si possono utilizzare)
    parser.add_argument('--save_images', type = bool, default = True, help = "If true, output images will be saved")
    parser.add_argument('--save_codes', type = bool, default = True, help = "If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type = bool, default = True, help = "If true, output meshes will be saved")
    args = parser.parse_args()

    face_rec_model, conf = load_model(args.path_to_models, args.model_name) # Viene caricato il modello di ricostruzione facciale e spostato su GPU
    face_rec_model.cuda()
    face_rec_model.eval()  

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    for subject_folder in input_folder.iterdir(): # Per ogni cartella contenuta nella cartella "input_folder" (sarebbe ogni soggetto, in pratica sarebbe ognuna delle cartelle con un nome randomico)
        if subject_folder.is_dir(): # Si verificare che subject_folder sia effettivamente una cartella
            for au_folder in subject_folder.iterdir(): # A questo punto, si itera su ciascuna delle sotto - cartella, su ciascuna delle cartella corrispondenti ad un action unit
                if au_folder.is_dir():
                    for frames_folder in au_folder.iterdir(): # Infine, si analizzano i frames contenuti in ciascuna cartella frames_1/frames_2/frames_3... 
                        if frames_folder.is_dir():
                            
                            relative_path = frames_folder.relative_to(input_folder)
                            output_folder = Path(args.output_folder) / relative_path  
                            
                            if output_folder.exists() and any(output_folder.iterdir()):
                                print(f"Folder {output_folder} already exists and is not empty. Skipping.")
                                continue
                            
                            output_folder.mkdir(parents = True, exist_ok = True)
                            print(f"Processing folder {frames_folder}")
                            process_folder(frames_folder, output_folder, face_rec_model, conf, args) # Viene chiamata la funzione "process_folder" per elaborare le immagini

    print("Done")
 
if __name__ == '__main__':
    main()
