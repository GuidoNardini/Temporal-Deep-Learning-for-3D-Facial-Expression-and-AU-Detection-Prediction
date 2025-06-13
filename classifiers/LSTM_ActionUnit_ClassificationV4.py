import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VIENE DEFINITA LA CLASSE DEL DATASET
class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, subject = None, max_seq_length = None):
        # root_dir: directory contenente i dati da caricare
        # subject: soggetto da considerare (se non specificato, vengono considerati tutti i soggetti)
        # max_seq_length: numero di features che deve avere ogni sequenza/video (nel caso dei file exp.npy si devono avere n*100 elementi per ogni sequenza/video, dove "n" è il numero di frames mentre 100 è il numero di features contenute nei file exp.npy)

        self.data = []  # Lista per contenere le sequenze dei valori contenuti nei file exp.npy, globalpose.npy... (a seconda di quali si vogliono utilizzare)
        self.labels = []  # Lista per contenere le etichette corrispondenti alle sequenze
        self.label_map = {}  # Dizionario per mappare le etichette (AU_1, AU_2...) in interi
        self.seq_lengths = [] # Lista necessaria per salvare il numero di frames originali (prima del padding) di ciascun video

        # Contatori per verificare il numero di video che vengono "paddati" perché hanno un numero di frames inferiore rispetto alla soglia e del numero di video che non vengono paddati
        self.padding_counter = 0
        self.no_padding_counter = 0

        self.processed_subjects = [] # Lista vuota che conterrà i nomi dei soggetti analizzati, in modo da avere uno storico dell'ordine di processamento
        
        self.subjects = [subject] if subject else os.listdir(root_dir)
        # Se vengono specificati uno o più soggetti vengono analizzati solo quelli, altrimenti vengono considerati tutti i soggetti presenti nella directory
    
        self.max_seq_length = max_seq_length 

        # CICLO PER OGNI SOGGETTO (CARTELLE CON NOMI RANDOM)
        for subject in self.subjects:
            subject_path = os.path.join(root_dir, subject)
            print(subject_path)
            if not os.path.isdir(subject_path):
                raise ValueError(f"Il soggetto {subject} non esiste in {root_dir}")

            self.processed_subjects.append(os.path.basename(subject_path)) # Il path del soggetto in analisi viene inserito nello storico dei soggetti analizzati
            
            # CICLO PER OGNI ACTION UNIT (AU_1, AU_2...) DEL SOGGETTO IN ANALISI
            excluded_AUs = {} # "AU_43", "AU_45" Inserire eventuali AU da non considerare nell'analisi, ad esempio quelle relative agli occhi
            
            for au_folder in sorted(os.listdir(subject_path), key = lambda x: int(''.join(filter(str.isdigit, x)))): # Ordina le cartelle in base al numero dell'Action Unit
                if au_folder in excluded_AUs:
                    continue
                au_path = os.path.join(subject_path, au_folder)
                print(au_path)
                if not os.path.isdir(au_path):
                    continue
                
                # CICLO PER OGNI CARTELLA "FRAMES" DELL'ACTION UNIT
                for frames_folder in sorted(os.listdir(au_path), key = lambda x: int(''.join(filter(str.isdigit, x)))):
                    frames_path = os.path.join(au_path, frames_folder)
                    print(frames_path)
                    if not os.path.isdir(frames_path):
                        continue
    
                    frames = []  # Lista vuota per memorizzare il contenuto dei file exp.npy di ciascun frame
                    # CICLO PER OGNI FRAME CONTENUTO NELLA CARTELLA "FRAMES" DELL'ACTION UNIT
                    for frame_folder in sorted(os.listdir(frames_path), key = lambda x: int(''.join(filter(str.isdigit, x)))): # Ordina le cartelle in base al numero del frame (affinché i frame siano ordinati temporalmente)
                        
                        # Le righe di codice sottostanti permettono di caricare i file di interesse
                        # Attenzione! Nel main ricordarsi di cambiare il numero delle features in ingresso in funzione di quali file si decide di caricare
                        frame_path_exp = os.path.join(frames_path, frame_folder, "exp.npy")
                        frame_path_globalpose = os.path.join(frames_path, frame_folder, "globalpose.npy")
                        frame_path_jawpose = os.path.join(frames_path, frame_folder, "jawpose.npy")
                        #frame_path_shape = os.path.join(frames_path, frame_folder, "shape.npy")
                        
                        exp_data = np.load(frame_path_exp)
                        globalpose_data = np.load(frame_path_globalpose)
                        jawpose_data = np.load(frame_path_jawpose)
                        #shape_data = np.load(frame_path_shape)     

                        frame_data = np.concatenate((exp_data, globalpose_data, jawpose_data))
                        frames.append(frame_data)      
                          
                    original_seq_len = len(frames) # Viene ottenuto il numero di frames del video in analisi

                    frames = np.stack(frames) # I frames vengono impilati preservando la loro struttura bidimensionale; quindi, per ogni video, si ottiene un array [n_frames, n_features]
                    #print(frames.shape) # Ad esempio, se un video ha 38 frames, si ottiene frames.shape = (38, 106), dove 106 è il numero di frames se si considerano exp.npy, jawpose.npy e globalpose.npy

                    if self.max_seq_length: # Se "max_seq_length" è specificato   
                        if original_seq_len < self.max_seq_length: # Viene fatto il padding fino alla lunghezza massima andando ad aggiungere zeri fino a raggiungere la lunghezza desiderata, ovvero max_seq_length
                            pad_length = self.max_seq_length - original_seq_len
                            frames = np.pad(frames, ((0, pad_length), (0,0)), mode = 'constant', constant_values = 0)
                            #print(frames)
                            #print(len(frames))
                            #print(len(frames[0]))
                            self.seq_lengths.append(original_seq_len)
                            self.padding_counter += 1
                        else: # Se invece il numero dei frames è maggiore della lunghezza massima allora vengono troncati i frames in eccesso
                            frames = frames[-self.max_seq_length:] # frames[:self.max_seq_length] permette di partire dall'inizio
                            self.seq_lengths.append(self.max_seq_length)
                            self.no_padding_counter += 1
                    else:
                        self.seq_lengths.append(original_seq_len)
                        
                    # La sequenza dei valori di un video vengono aggiunti al dataset
                    frames = torch.tensor(frames, dtype = torch.float32, device = device) # Ogni video avrà una forma [n_frames, n_features]
                    self.data.append(frames)

                    # Le righe di codice sottostanti servono per mappare le etichette (AU_1, AU_2...) in numeri interi, per una semplicità di utilizzo
                    if au_folder not in self.label_map:
                        self.label_map[au_folder] = len(self.label_map)
                    self.labels.append(self.label_map[au_folder])

        self.data = torch.stack(self.data)  # La lista di tensori viene concatenata in un unico tensore PyTorch di dimensioni [n_video, n_frames, n_features]
        # Alla fine del ciclo, "self.data" sarà un tensore tridimensionale [n_video, n_frames, n_features]
        
        self.labels = torch.tensor(self.labels, dtype = torch.long, device = device)  # La lista delle etichette viene convertita in un tensore PyTorch
        self.reverse_label_map = {v: k for k, v in self.label_map.items()} # Dizionario per mappare i numeri interi in etichette (AU_1, AU_2...)
 
        print(f"Numero di video paddati: {self.padding_counter}")
        print(f"Numero di video non paddati: {self.no_padding_counter}")
        
        '''
        print(self.data)
        print(self.data.shape) # torch.Size([2715, 60, 106])
        print(type(self.data))
        print(self.labels)
        print(len(self.labels)) # len(self.labels) = 2715
        print(self.label_map)
        print(self.data[0])
        print(len(self.data[0]))
        print(self.labels[0])
        '''
    
    def __len__(self):
        return len(self.data)  # Restituisce la lunghezza del dataset

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.seq_lengths[idx]  # Restituisce il dato e l'etichetta per un dato indice

    def get_processed_subjects(self): # Restituisce l'ordine dei soggetti processati
        return self.processed_subjects

# La funzione collate_fn serve a personalizzare il modo in cui i campioni del dataset vengono raggrupati in un batch durante il caricamento con un DataLoader
def collate_fn(batch):
    sequences, labels, seq_lengths = zip(*batch)
    sequences = torch.stack(sequences)
    labels = torch.tensor(labels, dtype = torch.long)
    seq_lengths = torch.tensor(seq_lengths, dtype = torch.long) # Converte le lunghezze originali (prima del padding) in un tensore 1D, importante per il packing delle sequenze nell'LSTM
    return sequences, labels, seq_lengths


# VIENE DEFINITA LA CLASSE DEL MODELLO UTILIZZANDO UN'ARCHITETTURA LSTM (Long Short - Term Memory)
class FacialExpressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FacialExpressionModel, self).__init__()
        # "input_size" è il numero di features per ciascun video di input (es. 100*40 nel caso si utilizzino i file exp.npy)
        # "hidden_size" è il numero di unità nascoste nel LSTM (inluenza la capacità del modello di memorizzare, più è alto il valore più precisione si ottiene ma cè un maggiore rischio di overfitting)
        # "num_classes" è il numero di classi da predire (es. 24 nel caso delle AU)
          
        # Layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, bidirectional = True)  # Layer che è in grado di imparare informazioni su sequenze temporali
        # batch_first = True significa che l'input e l'output avranno la stessa forma, ovvero (batch_size, sequence_length, input_size)

        # Layer Fully Connected (FC)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # Viene ridotta la dimensione dell'output del LSTM a 128
        self.fc2 = nn.Linear(128, 64)  # Viene ridotta la dimensione dell'output del FC1 a 64
        self.fc3 = nn.Linear(64, num_classes)  # Viene ridotta la dimensione dell'output del FC2 al numero di classi da predire

        self.relu = nn.ReLU()  # Funzione di attivazione ReLU
        #self.softmax = nn.Softmax(dim = 1)  # Funzione softmax per la classificazione multi-classe
        self.dropout = nn.Dropout(0.5)  # Dropout per evitare l'overfitting (durante il training, a ogni passo, vengono disattivate alcune unità in modo casuale)

    def forward(self, x, seq_lengths):       
        #print("Input sequence", x.shape) # x.shape = torch.Size([batch_size, n_frames, n_features_for_frame])
        
        # La riga di codice sottostante permette di convertire il tensore x (con padding) in una sequenza PackedSequence, ottimizzata per l'elaborazione di sequenze di lunghezza variabile in PyTorch
        # Quindi:
        # - x è un tensore di input con padding, di forma (batch_size, max_seq_length, num_features)
        # - seq_lengths sono le lunghezze originali delle sequenze (prima del padding)
        # In particolare, la riga di codice sottostante permette di ignorare i valori di padding durante i calcoli dell'LSTM
        packed_x = rnn_utils.pack_padded_sequence(x, seq_lengths.cpu(), batch_first = True, enforce_sorted = False) 
        packed_out, _ = self.lstm(packed_x) # l'LSTM elabora le sequenze e successivamente viene riaggiunto il padding per le elaborazioni successive...
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first = True) # Questa riga di codice è necessaria perché i layer FC non possono elaborare oggetti di tipo PackedSequence
        last_outputs = out[torch.arange(out.size(0)), seq_lengths - 1]

        # Forward pass attraverso il primo layer FC
        x = self.fc1(last_outputs)
        x = self.relu(x)
        x = self.dropout(x)
        # Forward pass attraverso il secondo layer FC
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Forward pass attraverso il terzo layer FC
        x = self.fc3(x)
        #return self.softmax(x)  # Restituisce l'output della rete
        return x

class EarlyStopping:
    def __init__(self, patience = 5, delta = 0.01):
        self.patience = patience # numero di epoche da attendere prima di fermare il training
        self.delta = delta # soglia di tolleranza
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def split_train_test_data(data, labels, test_size = 0.2):  
    
    '''
    Funzione per suddividere il dataset in set di training e test mantenendo la struttura dei soggetti.
    L'obiettivo è evitare che video dello stesso soggetto finiscano alcuni nel set di training e altri in quello di test.

    Argomenti:
    - data: tensore PyTorch contenente i dati da suddividere;
    - labels: tensore PyTorch contenente le etichette corrispondenti ai dati;
    - test_size: frazione del dataset da assegnare al test set (default: 20%)

    Ritorna:
    - train_data: tensore PyTorch con i dati del set di training;
    - test_data: tensore PyTorch con i dati del set di test;
    - train_labels: tensore PyTorch con le etichette del set di training;
    - test_labels: tensore PyTorch con le etichette del set di test.

    Funzionamento:
    labels è fatto in questo modo: tensor([ 0,  0,  1,  ..., 22, 23, 23, 0, 0, 1, 2, 2...23, 23])
    Ogni volta che c'è uno zero significa che è un nuovo soggetto, ecco come vengono divisi i dati nel dataset.
    Ovviamente se ci sono due zeri vicino, si considera il primo...questo perché ciascun soggetto ha due video della AU_1.
    Quindi, una volta trovata la posizione di tutti gli zeri è possibile tagliare opportunatamente i tensori dei dati e delle etichette.
    '''

    zero_indices = torch.where(labels == 0)[0]
    unique_subject_indices = [zero_indices[0].item()]
    for i in range(1, len(zero_indices)):
        if zero_indices[i] != zero_indices[i-1] + 1:
            unique_subject_indices.append(zero_indices[i].item())
    unique_subject_indices.append(len(labels))
    
    num_subjects = len(unique_subject_indices) - 1
    num_train_subjects = int(round(num_subjects * (1 - test_size)))
    
    train_start = unique_subject_indices[0]
    train_end = unique_subject_indices[num_train_subjects]
    test_start = unique_subject_indices[num_train_subjects]
    test_end = unique_subject_indices[-1]
    
    train_indices = torch.arange(train_start, train_end)
    test_indices = torch.arange(test_start, test_end)
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    return train_data, test_data, train_labels, test_labels, train_indices, test_indices

def main():
    root_dir = '/andromeda/personal/fgastaldi/facemorphic_output'  # CAMBIARE CON LA DIRECTORY CORRETTA DEL DATASET DI RIFERIMENTO
    max_seq_length = 20  # Numero di frame da prelevare per ciascun video (sarebbe la lunghezza massima per ciascuna sequenza!)
    # 20 frames: 55.64% (210 video paddati e 2505 video non paddati)
    # 30 frames: 51.43% (364 video paddati e 2351 video non paddati)
    # 40 frames: 59.46% (843 video paddati e 1873 video non paddati)
    # 50 frames: 55.45% (1453 video paddati e 1262 video non paddati)
    # 60 frames: 55.26% (1989 video paddati e 726 video non paddati)
    # 70 frames: 57.55% (2352 video paddati e 363 video non paddati)
    # 80 frames: 55.64% (2574 video paddati e 141 video non paddati)

    dataset = FacialExpressionDataset(root_dir, max_seq_length = max_seq_length) # Viene creato il dataset
    #print(dataset.get_processed_subjects()) # Viene stampato l'ordine con cui i soggetti sono stati processati
    # PRIMO SOGGETTO ANALIZZATO: 74bf78720495496ca29abfc78e527b42
    # ULTIMO SOGGETTO TRAINING_SET (46-esimo): f2eac7b475464a48b1cbb866db012ed1
    # PRIMO SOGGETTO TEST_SET (47-esimo): af0b27f1d96e410a9c600d671f57e4e8

    train_data, test_data, train_labels, test_labels, train_indices, test_indices = split_train_test_data(dataset.data, dataset.labels)

    '''
    print(dataset.data[0].shape[1])
    print(dataset.data[0])
    print(train_data)
    print(train_data.shape)
    print(test_data)
    print(test_data.shape)
    print(train_labels)
    print(train_labels.shape)
    print(test_labels)
    print(test_labels.shape)
    '''

    seq_lengths = torch.tensor(dataset.seq_lengths)
    train_seq_lengths = seq_lengths[train_indices]
    test_seq_lengths = seq_lengths[test_indices]

    train_labels = train_labels.clone().detach()
    test_labels = test_labels.clone().detach()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels, train_seq_lengths)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels, test_seq_lengths)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, collate_fn = collate_fn)

    input_size = dataset.data[0].shape[1]  # Numero di features per frames
    hidden_size = 128  # Numero di unità nel LSTM
    num_classes = len(dataset.label_map) # num_classes = 24, pari al numero di AU

    model = FacialExpressionModel(input_size, hidden_size, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, factor = 0.1)
    early_stopping = EarlyStopping(patience = 5, delta = 0.01)

    # TRAINING
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, seq_lengths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, seq_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(running_loss / len(train_loader))

        if early_stopping(running_loss / len(train_loader)):
            print("Early stopping")
            break

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    print("Training complete!")

    # TESTING
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, seq_lengths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, seq_lengths)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test data: {round(100 * correct / total, 2)} %")


    # MATRICE DI CONFUSIONE
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize = (10, 10))
    sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = dataset.reverse_label_map.values(), yticklabels = dataset.reverse_label_map.values())
    plt.xlabel('LABELS PREDICTED')
    plt.ylabel('LABELS TRUE')    
    plt.title('CONFUSION MATRIX')
    plt.savefig('confusion_matrixV4.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()

