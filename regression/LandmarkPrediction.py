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

class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, subjects, sequence_length = 20, transform = None):
 
        self.root_dir = root_dir # Directory principale in cui si trovano i dati per creare il dataset
        self.subjects = subjects # Lista dei soggetti da includere nel dataset (saranno differenti per training e test)
        self.sequence_length = sequence_length # Numero di frames RGB da utilizzare per formare una sequenza (default 20)
        self.transform = transform # Eventuali trasformazioni da applicare alle immagini (da notare che sono già state croppate a 224x224)
        self.data = self._load_data() # self.data è una lista di tuple (images, npy_0, exp_target) in cui:
        # - images è un tensore di dimensioni [N, C, H, W] (N = numero di frames, C = numero di canali, H = altezza, W = larghezza)
        # - npy_0 è un tensore di dimensioni [exp_dim + globalpose_dim + jawpose_dim] che rappresenta l'espressione del primo frame
        # - npy_target è un tensore di dimensioni [exp_dim + globalpose_dim + jawpose_dim] che rappresenta l'espressione del frame N+1
        # La funzione _load_data() carica i dati e restituisce una lista di tuple (immagini, npy_0, exp_target)

    def _load_data(self):
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
                    
                    exp_target = np.load(os.path.join(frame_paths[self.sequence_length + 1], "exp.npy")) # Carica il file exp.npy del frame N+1 come target finale
                    globalpose_target = np.load(os.path.join(frame_paths[self.sequence_length + 1], "globalpose.npy")) # Carica il file globalpose.npy del frame N+1 come target finale
                    jawpose_target = np.load(os.path.join(frame_paths[self.sequence_length + 1], "jawpose.npy")) # Carica il file jawpose.npy del frame N+1 come target finale
                    npy_target = np.concatenate((exp_target, globalpose_target, jawpose_target), axis = 0) # Concatena i tre file .npy in un unico array

                    sequences.append((images, torch.tensor(npy_0, dtype = torch.float32), torch.tensor(npy_target, dtype = torch.float32))) # Aggiunge la sequenza alla lista
        
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, npy_0, npy_target = self.data[idx]
        return images, npy_0, npy_target

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
        npy_pred = self.fc_out(lstm_out[:, -1, :]) # Prende l'output finale dell'LSTM (l'ultimo stato nascosto) e lo passa attraverso un layer fully connected per ottenere la previsione finale
        return npy_pred

def accuracy_within_threshold(npy_pred, npy_target, threshold = 0.1):
    diff = torch.abs(npy_pred - npy_target)
    correct = (diff < threshold).float()
    return correct.mean().item()

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

    sequence_length = 20 
    # Vengono creati i dataset di train e di test utilizzando la classe "FacialExpressionDataset"
    train_dataset = FacialExpressionDataset(dataset_path, train_subjects, sequence_length = sequence_length, transform = transform)
    #print(f"Number of train sequences: {len(train_dataset)}")
    test_dataset = FacialExpressionDataset(dataset_path, test_subjects, sequence_length = sequence_length, transform = transform)
    #print(f"Number of test sequences: {len(test_dataset)}")

    # Vengono creati i DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    print(f"Number of train batches: {len(train_loader)}")
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    print(f"Number of test batches: {len(test_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetLSTM().to(device)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    num_epochs = 50
    patience = 5
    patience_counter = 0
    best_test_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc = "Epoch Progress", unit = "epoch"):

        # CICLO DI TRAINING
        model.train() 
        running_loss = 0.0
        for images, npy_0, npy_target in train_loader: 
            images, npy_0, npy_target = images.to(device), npy_0.to(device), npy_target.to(device)   
            optimizer.zero_grad() 
            npy_pred = model(images, npy_0)
            loss = criterion(npy_pred, npy_target) 
            loss.backward()  
            optimizer.step()     
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

        # CICLO DI TEST
        model.eval() 
        test_loss = 0.0
        acc_total = 0.0
        with torch.no_grad(): 
            for images, npy_0, npy_target in test_loader:
                images, npy_0, npy_target = images.to(device), npy_0.to(device), npy_target.to(device)
                # images.shape = [batch_size, sequence_length, 3, 224, 224]
                # npy_0.shape = [batch_size, exp_dim + globalpose_dim + jawpose_dim]
                # npy_target.shape = [batch_size, exp_dim + globalpose_dim + jawpose_dim]
                
                npy_pred = model(images, npy_0)
                           
                # Calcolo della loss
                loss = criterion(npy_pred, npy_target)
                test_loss += loss.item()
                # Calcolo dell'accuratezza
                acc = accuracy_within_threshold(npy_pred, npy_target, threshold = 0.1)
                acc_total += acc
        avg_test_loss = test_loss / len(test_loader)
        avg_acc = acc_total / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy (threshold = 0.1): {avg_acc*100:.2f}%")

        # EARLY STOPPING
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            print(f"New best test loss: {best_test_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()