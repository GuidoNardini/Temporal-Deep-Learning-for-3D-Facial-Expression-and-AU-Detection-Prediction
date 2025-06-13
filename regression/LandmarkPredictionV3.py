import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, subjects, sequence_length = 20, transform = None, cache_file = None):
 
        self.root_dir = root_dir # Directory principale in cui si trovano i dati per creare il dataset
        self.subjects = subjects # Lista dei soggetti da includere nel dataset (saranno differenti per training e test)
        self.sequence_length = sequence_length # Numero di frames RGB da utilizzare per formare una sequenza (default 20)
        self.transform = transform # Eventuali trasformazioni da applicare alle immagini (da notare che sono già state croppate a 224x224)
        self.cache_file = cache_file

        if self.cache_file and os.path.exists(self.cache_file):
            print(f"Loading dataset from cache: {self.cache_file}")
            self.data = torch.load(self.cache_file)
        
        else:
            self.data = self.load_data() # self.data è una lista di tuple (images, npy_0, exp_target) in cui:
            # - images è un tensore di dimensioni [N, C, H, W] (N = numero di frames, C = numero di canali, H = altezza, W = larghezza)
            # - npy_0 è un tensore di dimensioni [exp_dim + globalpose_dim + jawpose_dim] che rappresenta l'espressione del primo frame
            # - npy_target è un tensore di dimensioni [exp_dim + globalpose_dim + jawpose_dim] che rappresenta l'espressione del frame N+1
            # La funzione load_data() carica i dati e restituisce una lista di tuple (immagini, npy_0, exp_target)
            if self.cache_file:
                print(f"Saving dataset to cache: {self.cache_file}")
                torch.save(self.data, self.cache_file)

    def load_data(self):
        sequences = [] #Lista per memorizzare le sequenze di immagini e i file .npy d'interesse
        
        for subject in tqdm(self.subjects, desc = "Loading subjects"): # Ciclo sui soggetti
            subject_path = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            
            for action_unit in sorted(os.listdir(subject_path), key = lambda x: int(''.join(filter(str.isdigit, x)))): # Ciclo sulle Action Units
                action_path = os.path.join(subject_path, action_unit)
                if not os.path.isdir(action_path):
                    continue
                
                for video in sorted(os.listdir(action_path), key = lambda x: int(''.join(filter(str.isdigit, x)))): # Ciclo sui video
                    video_path = os.path.join(action_path, video)
                    if not os.path.isdir(video_path):
                        continue
                    
                    frames = sorted(os.listdir(video_path), key = lambda x: int(''.join(filter(str.isdigit, x)))) # I frames del video in analisi vengono ordinati in base al numero nel nome del file
                    # Se un video ha meno di 22 frame, lo si salta...
                    # Del frame_0 viene prelevato il file .npy, dei frame_1 - frame_20 vengono prese le immagini RGB, mentre del frame_21 viene prelevato il file .npy (in tutto sono 22 frames)
                    if len(frames) < self.sequence_length + 2:
                        print(f"Skipping video {video_path} due to insufficient frames.")
                        continue
                      
                    # La riga di codice sottostante crea una lista (della lunghezza desiderata) dei paths dei frames ordinati
                    frame_paths = sorted([os.path.join(video_path, frames[j]) for j in range(self.sequence_length + 2)], key = lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
                        
                    images = [] # Lista per raccogliere le immagini
                    npy_0 = None
                    npy_target = None

                    for path in frame_paths[1:self.sequence_length + 1]: # Ciclo sui frame, vengono esclusi il primo e l'ultimo (da cui si prenderanno i file .npy)
                        img_path = os.path.join(path, "inputs.png") # Path dell'immagine RGB
                        image = Image.open(img_path) # Carica l'immagine
                        if self.transform:
                            image = self.transform(image) # Applica le trasformazioni (se definite)
                        images.append(image)
                    
                    images = torch.stack(images) # Converte la lista di immagini in un tensore
                    
                    exp_0 = np.load(os.path.join(frame_paths[0], "exp.npy")) # Carica il file exp.npy del frame 0 come input iniziale
                    globalpose_0 = np.load(os.path.join(frame_paths[0], "globalpose.npy")) # Carica il file globalpose.npy del frame 0 come input iniziale
                    jawpose_0 = np.load(os.path.join(frame_paths[0], "jawpose.npy")) # Carica il file jawpose.npy del frame 0 come input iniziale 
                    npy_0 = np.concatenate((exp_0, globalpose_0, jawpose_0), axis = 0) # Concatena i tre file .npy in un unico array
                    
                    target_frame_path = frame_paths[self.sequence_length + 1] # Questa riga di codice permette di salvare nel dataset anche i path da cui vengono presi i file .npy, in modo tale da poterci salvare quelli predetti dal modello
                    exp_target = np.load(os.path.join(target_frame_path, "exp.npy")) # Carica il file exp.npy del frame N+1 come target finale
                    globalpose_target = np.load(os.path.join(target_frame_path, "globalpose.npy")) # Carica il file globalpose.npy del frame N+1 come target finale
                    jawpose_target = np.load(os.path.join(target_frame_path, "jawpose.npy")) # Carica il file jawpose.npy del frame N+1 come target finale
                    npy_target = np.concatenate((exp_target, globalpose_target, jawpose_target), axis = 0) # Concatena i tre file .npy in un unico array

                    sequences.append((images, torch.tensor(npy_0, dtype = torch.float32), torch.tensor(npy_target, dtype = torch.float32), target_frame_path)) # Aggiunge la sequenza alla lista
        
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, npy_0, npy_target, target_frame_path = self.data[idx]
        return images, npy_0, npy_target, target_frame_path

# Il modello ha l'obiettivo di prevedere il vettore .npy del frame N a partire da una sequenza di immagini e dal vettore .npy del frame 0
# In particolare:
# 1) Riceve in input una sequenza di immagini (20 frames RGB) e il file .npy del frame 0
# 2) Estrae un embedding da ciascuna delle immagini utilizzando un modello ResNet18 pre - addestrato
# 3) Proietta il file .npy del frame 0 in uno spazio di dimensione 128
# 4) Processa la sequenza di embedding tramite un LSTM, inizializzato con uno stato nascosto (h_0) e una cella di memoria (c_0) derivati dal file .npy del frame 0 (proiettato nel nuovo spazio)
# 5) Restituisce il file .npy del frame N, che rappresenta l'output finale del modello
class ResNetLSTM(nn.Module):
    def __init__(self, embedding_dim = 512, lstm_hidden_dim = 128, npy_dim = 106, projection_dim = 128, num_lstm_layers = 1):
        super(ResNetLSTM, self).__init__()
        
        resnet18 = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        resnet18.fc = nn.Identity() 
        self.feature_extractor = resnet18
        
        self.embedding_dim = embedding_dim
        self.npy_projection = nn.Linear(npy_dim, projection_dim)
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = lstm_hidden_dim, num_layers = num_lstm_layers, batch_first = True)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(lstm_hidden_dim, npy_dim)

    def forward(self, images, npy_0):
        batch_size, seq_len, C, H, W = images.shape # images.shape = [batch_size, seq_len, C, H, W]
        images = images.view(batch_size * seq_len, C, H, W) # Il tensore delle immagini viene riformattato per passarlo alla ResNet, tutte le immagini della sequenza vengono concatenate in un batch unico
        
        embeddings = self.feature_extractor(images) # Estrae gli embeddings da ciascuna immagine, l'output avrà forma [batch_size * seq_len, embedding_dim]
        embeddings = embeddings.view(batch_size, seq_len, self.embedding_dim) # Riformatta l'output della ResNet in modo che abbia forma [batch_size, seq_len, embedding_dim] per essere input dell'LSTM

        npy_0_proj = self.npy_projection(npy_0) # Proietta il file .npy del frame 0 in uno spazio di dimensione 128
        
        h_0 = npy_0_proj.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1) # Aggiunge una dimensione per il numero di layer dell'LSTM e ripete il tensore per il numero di layer. Inizializza lo stato nascosto dell'LSTM con il file .npy del frame 0 proiettato
        c_0 = torch.zeros_like(h_0) 
        lstm_out, _ = self.lstm(embeddings, (h_0, c_0)) # Passa la sequenza di embedding all'LSTM
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        npy_pred = self.fc_out(lstm_out) # Prende l'output finale dell'LSTM (l'ultimo stato nascosto) e lo passa attraverso un layer fully connected per ottenere la previsione finale
        return npy_pred

def accuracy_within_threshold(npy_pred, npy_target, threshold = 0.05):
    # npy_pred e npy_target sono tensori di dimensioni [8, 106]
    # diff è un tensore di dimensioni [8, 106] con la differenza dei valori tra predetti e target
    # correct è un tensore di dimensioni [8,  106] che contiene 1 quando la differenza è sotto soglia, 0 altrimenti
    # Viene restituita la percentuale di componenti predette "giuste" entro la soglia
    diff = torch.abs(npy_pred - npy_target)
    correct = (diff < threshold).float()
    return correct.mean().item()

def plot_au_mean_curves(au_results):
    
    for au_name, data in au_results.items():
        preds = np.array(data["preds"]) 
        targets = np.array(data["targets"]) 

        exp_preds = preds[:, :100]
        globalpose_preds = preds[:, 100:103]
        jawpose_preds = preds[:, 103:]

        exp_targets = targets[:, :100]
        globalpose_targets = targets[:, 100:103]
        jawpose_targets = targets[:, 103:]

        # Errore MAE
        exp_error = np.abs(exp_preds - exp_targets).mean(axis = 0)
        globalpose_error = np.abs(globalpose_preds - globalpose_targets).mean(axis = 0)
        jawpose_error = np.abs(jawpose_preds - jawpose_targets).mean(axis = 0)
        
        plt.figure(figsize = (10, 4))
        plt.plot(exp_error, label = "exp - MAE", color = "green")
        plt.axhline(y = 0.05, color = 'red', linestyle = '--', label = 'Threshold = 0.05')
        plt.title(f"{au_name} - Mean Absolute Error - Expression Params")
        plt.xlabel("Component index")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/home/fgastaldi/regression/AU_plots/{au_name}_exp_mae_plot.png")
        plt.close()

        plt.figure(figsize = (10, 4))
        plt.plot(globalpose_error, label = "globalpose - MAE", color = "green")
        plt.axhline(y = 0.05, color = 'red', linestyle = '--', label = 'Threshold = 0.05')
        plt.title(f"{au_name} - Mean Absolute Error - Globalpose params")
        plt.xlabel("Component index")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/home/fgastaldi/regression/AU_plots/{au_name}_globalpose_mae_plot.png")
        plt.close()

        plt.figure(figsize = (10, 4))
        plt.plot(jawpose_error, label = "jawpose - MAE", color = "green")
        plt.axhline(y = 0.05, color = 'red', linestyle = '--', label = 'Threshold = 0.05')
        plt.title(f"{au_name} - Mean Absolute Error - Jawpose params")
        plt.xlabel("Component index")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/home/fgastaldi/regression/AU_plots/{au_name}_jawpose_mae_plot.png")
        plt.close()

def main():
    
    # Trasformazioni da applicare alle immagini
    # Si ricorda che le immagini sono già state croppate a 224x224
    transform = transforms.Compose([
        transforms.ToTensor()  
    ])

    # Percorso da cui prelevare i dati
    dataset_path = "/andromeda/personal/fgastaldi/facemorphic_output"

    # Le righe di codice sottostanti servono per dividere i soggetti in training e test
    # In particolare, il 20% dei soggetti viene utilizzato per il test e l'80% per il training
    all_subjects = os.listdir(dataset_path)
    split_idx = int(0.8 * len(all_subjects))
    train_subjects = all_subjects[:split_idx]
    #print(f"Number of train subjects: {len(train_subjects)}")
    test_subjects = all_subjects[split_idx:]
    #print(f"Number of test subjects: {len(test_subjects)}")
    #test_subjects = ['f2eac7b475464a48b1cbb866db012ed1', 'af0b27f1d96e410a9c600d671f57e4e8', 'f2ec6a25ceed4cafb2589b2796e37bf3', '6e62e6154a6744d5b48f99b36cbf3915', '15a609ca0bcd43748443c1b56256a655', '9cd270403b024c34b651f9fe20feff2f', 'a6d96edbd8394958a1836b33362ccfa7', '5533380ff8ef4d988ac8930271e0eac0', '152d842abec94101bdeb6bb0744849b3', '028c231cd32a48e19613fe653680c11e', 'c58d1e346b564de6be3dbe40dd046e7d', '7f079b0ac0384b7b9f0977c54b4411e1']

    sequence_length = 20
    # Vengono creati i dataset di train e di test utilizzando la classe "FacialExpressionDataset"
    train_cache_file = "train_dataset.pt"
    train_dataset = FacialExpressionDataset(dataset_path, train_subjects, sequence_length = sequence_length, transform = transform, cache_file = train_cache_file)
    #print(f"Number of train sequences: {len(train_dataset)}")
    test_cache_file = "test_dataset.pt"
    test_dataset = FacialExpressionDataset(dataset_path, test_subjects, sequence_length = sequence_length, transform = transform, cache_file = test_cache_file)
    #print(f"Number of test sequences: {len(test_dataset)}")
    
    # Vengono creati i DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    #print(f"Number of train batches: {len(train_loader)}")
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    #print(f"Number of test batches: {len(test_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetLSTM().to(device)

    for param in model.feature_extractor.parameters(): # Vengono sbloccati tutti i pesi della ResNet18 affinché possano essere aggiornati durante il training
        param.requires_grad = True


    # Vengono inizializzate due liste per contenere i parametri della ResNet18 e dell'LSTM
    resnet_params = []
    lstm_params = []
    for name, param in model.named_parameters():
        if "feature_extractor" in name: # Se il nome del parametro contiene "feature_extractor" significa che appartiene alla ResNet18
            resnet_params.append(param)
        else:
            lstm_params.append(param)

    criterion = nn.MSELoss() 

    optimizer_resnet = optim.Adam(resnet_params, lr = 0.0001) # Lavora solamente sui pesi della ResNet18
    optimizer_lstm = optim.Adam(lstm_params, lr = 0.001) # Lavora su tutti gli altri pesi (LSTM e fc_out)

    scheduler_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_resnet, mode = 'min', patience = 3, factor = 0.1)
    scheduler_lstm = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode = 'min', patience = 3, factor = 0.1)

    num_epochs = 50
    patience = 4
    patience_counter = 0
    best_test_loss = float('inf')

    train_losses = [] # Vengono inizializzate due liste per salvare le losses di training e test per ciascuna epoca (necessario per plottare la curva Training/Test Loss)
    test_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # CICLO DI TRAINING
        model.train() 
        running_loss = 0.0
        for images, npy_0, npy_target, target_paths in train_loader: 
            images, npy_0, npy_target = images.to(device), npy_0.to(device), npy_target.to(device)   
            
            optimizer_resnet.zero_grad()
            optimizer_lstm.zero_grad()

            npy_pred = model(images, npy_0)
            loss = criterion(npy_pred, npy_target) 

            loss.backward()  

            optimizer_resnet.step()  
            optimizer_lstm.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        scheduler_resnet.step(avg_train_loss)
        scheduler_lstm.step(avg_train_loss)

        print(f"Training Loss: {avg_train_loss:.4f}")

        # CICLO DI TEST
        model.eval() 
        test_loss = 0.0

        acc_exp_total = 0.0
        acc_globalpose_total = 0.0
        acc_jawpose_total = 0.0
        acc_total = 0.0

        all_predictions = []
        all_targets = []

        au_results = {}

        with torch.no_grad(): 
            for images, npy_0, npy_target, target_paths in test_loader:
                images, npy_0, npy_target = images.to(device), npy_0.to(device), npy_target.to(device)
                # images.shape = [batch_size, sequence_length, 3, 224, 224]
                # npy_0.shape = [batch_size, exp_dim + globalpose_dim + jawpose_dim]
                # npy_target.shape = [batch_size, exp_dim + globalpose_dim + jawpose_dim]
                
                npy_pred = model(images, npy_0)
                             
                # Calcolo della loss
                loss = criterion(npy_pred, npy_target)
                test_loss += loss.item()

                # Calcolo dell'accuratezza per ogni singolo file
                exp_pred = npy_pred[:, :100]
                globalpose_pred = npy_pred[:, 100:103]
                jawpose_pred = npy_pred[:, 103:]

                exp_target = npy_target[:, :100]
                globalpose_target = npy_target[:, 100:103]
                jawpose_target = npy_target[:, 103:]
 
                '''
                Range per exp.npy: Min = -4.051661014556885, Max = 5.044169902801514
                Range per globalpose.npy: Min = -1.0889180898666382, Max = 1.2578874826431274
                Range per jawpose.npy: Min = -0.20888470113277435, Max = 0.4943910837173462


                | Tipo             | Range totale   | Soglia MAE consigliata | Note |
                |------------------|----------------|-------------------------|------|
                | `exp.npy`        | ~9.1           | `0.2` - `0.4`           | 2-5% del range |
                | `globalpose.npy` | ~2.35          | `0.05` - `0.1`          | 2-4% del range |
                | `jawpose.npy`    | ~0.70          | `0.01` - `0.03`         | 2-5% del range |

                  '''
                
                acc_exp = accuracy_within_threshold(exp_pred, exp_target, threshold = 0.05)
                acc_globalpose = accuracy_within_threshold(globalpose_pred, globalpose_target, threshold = 0.05)
                acc_jawpose = accuracy_within_threshold(jawpose_pred, jawpose_target, threshold = 0.05)
                acc = accuracy_within_threshold(npy_pred, npy_target, threshold = 0.05) # Come funziona se le soglie sono differenti?? Va fatta una media delle soglie??
  
                acc_exp_total += acc_exp
                acc_globalpose_total += acc_globalpose
                acc_jawpose_total += acc_jawpose
                acc_total += acc

                all_predictions.append(npy_pred.cpu().numpy()) 
                all_targets.append(npy_target.cpu().numpy())            
                
                # Salvataggio delle predizioni
                # Salvataggio npy predetti e target per successivi plot

                for pred, target, path in zip(npy_pred, npy_target, target_paths):    
                    pred_np = pred.detach().cpu().numpy()
                    target_np = target.detach().cpu().numpy()

                    au_name = [folder for folder in path.split(os.sep) if folder.startswith("AU")][0]
                    if au_name not in au_results:
                        au_results[au_name] = {"preds": [], "targets": []}

                    au_results[au_name]["preds"].append(pred_np)
                    au_results[au_name]["targets"].append(target_np)

                    # Salvataggio predizioni
                    exp_pred = pred_np[:100]
                    globalpose_pred = pred_np[100:103]
                    jawpose_pred = pred_np[103:]
                    
                    save_exp_path = os.path.join(path, "exp_pred.npy")
                    save_globalpose_path = os.path.join(path, "globalpose_pred.npy")
                    save_jawpose_path = os.path.join(path, "jawpose_pred.npy")

                    np.save(save_exp_path, exp_pred)
                    np.save(save_globalpose_path, globalpose_pred)
                    np.save(save_jawpose_path, jawpose_pred)
                           
        all_predictions = np.concatenate(all_predictions, axis = 0)
        all_targets = np.concatenate(all_targets, axis = 0)

        avg_acc_exp = acc_exp_total / len(test_loader)
        avg_acc_globalpose = acc_globalpose_total / len(test_loader)
        avg_acc_jawpose = acc_jawpose_total / len(test_loader)
        avg_acc = acc_total / len(test_loader)

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy (exp, threshold = 0.05): {avg_acc_exp*100:.2f}%")
        print(f"Test Accuracy (globalpose, threshold = 0.05): {avg_acc_globalpose*100:.2f}%")
        print(f"Test Accuracy (jawpose, threshold = 0.05): {avg_acc_jawpose*100:.2f}%")       
        #print(f"Test Accuracy (threshold = 0.05): {avg_acc*100:.2f}%")

        # EARLY STOPPING
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            print(f"New best Test Loss: {best_test_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early Stopping Triggered.")
                break

    # CURVA DI TRAINING/TEST LOSS
    plt.figure(figsize = (10,6))
    plt.plot(train_losses, label = 'Training Loss', color = 'blue')
    plt.plot(test_losses, label = 'Test Loss', color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')  # Salva anche il grafico
    plt.show()

    # VENGONO REALIZZATI 3 GRAFICI PER CIASCUNA AUs
    # I VALORI VISUALIZZATI SONO LE MEDIE DEI 12 SOGGETTI PER CIASCUNA AUs
    plot_au_mean_curves(au_results)

if __name__ == "__main__":
    main()


