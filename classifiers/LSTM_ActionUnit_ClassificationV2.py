import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
        
        # Contatori per verificare il numero di video che vengono "paddati" perché hanno un numero di frames inferiore rispetto alla soglia e del numero di video che non vengono paddati
        self.padding_counter = 0
        self.no_padding_counter = 0

        self.processed_subjects = [] # Lista vuota che conterrà i nomi dei soggetti analizzati, in modo da avere uno storico dell'ordine di processamento
        self.subjects = []  # Lista dei soggetti da considerare
        if subject:
            self.subjects = [subject]
        else:
            # Se non è stato specificato un soggetto, vengono considerati tutti i soggetti presenti nella directory
            self.subjects = os.listdir(root_dir)
     
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
                        
                        #frames.append(exp_data)
                        
                    frames = np.concatenate(frames) # "frames" avrà lunghezza 100 (nel caso si utilizzi solo exp.npy, altrimenti la somma delle dimensioni dei vari file caricati) * numero di frame di ciascun video
                    #print(frames.shape)


                    if len(frames) > 0:
                        if self.max_seq_length: # Se "max_seq_length" è specificato        
                            if len(frames) < self.max_seq_length: # Viene fatto il padding fino alla lunghezza massima andando ad aggiungere zeri fino a raggiungere la lunghezza desiderata, ovvero max_seq_length
                                frames = np.pad(frames, (0, self.max_seq_length - len(frames)), mode = 'constant', constant_values = 0)
                                self.padding_counter += 1
                            else: # Se invece il numero dei frames è maggiore della lunghezza massima allora vengono troncati i frames in eccesso
                                frames = frames[-self.max_seq_length:] # frames[:self.max_seq_length] permette di partire dall'inizio
                                self.no_padding_counter += 1
                        
                        # La sequenza dei valori di un video vengono aggiunti al dataset
                        frames = torch.tensor(frames, dtype = torch.float32, device = device)
                        self.data.append(frames)

                        # Le righe di codice sottostanti servono per mappare le etichette (AU_1, AU_2...) in numeri interi, per una semplicità di utilizzo
                        if au_folder not in self.label_map:
                            self.label_map[au_folder] = len(self.label_map)
                        self.labels.append(self.label_map[au_folder])

        self.data = torch.stack(self.data)  # La lista dei tensori viene concatenata in un unico tensore PyTorch
        # Alla fine del ciclo for, "self.data" avrà dimensione: numero di video totali x numero di features per ciascun video...ovvero 40 (numero di frames che deve avere ciascun video) x 100 (numero di features nel caso di exp.npy)

        self.labels = torch.tensor(self.labels, dtype = torch.long, device = device)  # La lista delle etichette viene convertita in un tensore PyTorch
        self.reverse_label_map = {v: k for k, v in self.label_map.items()} # Dizionario per mappare i numeri interi in etichette (AU_1, AU_2...)
 
        print(f"Numero di video paddati: {self.padding_counter}")
        print(f"Numero di video non paddati: {self.no_padding_counter}")
        
        '''
        print(self.data)
        print(self.data.shape)
        print(type(self.data))
        print(self.labels)
        print(len(self.labels))
        print(self.label_map)
        print(self.data[0])
        print(len(self.data[0]))
        print(self.labels[0])
        '''
    
    def __len__(self):
        return len(self.data)  # Restituisce la lunghezza del dataset

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Restituisce il dato e l'etichetta per un dato indice

    def get_processed_subjects(self): # Restituisce l'ordine dei soggetti processati
        return self.processed_subjects

# VIENE DEFINITA LA CLASSE DEL MODELLO UTILIZZANDO UN'ARCHITETTURA LSTM (Long Short - Term Memory)
class FacialExpressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FacialExpressionModel, self).__init__()
        # "input_size" è il numero di features per ciascun video di input (es. 100*40 nel caso si utilizzino i file exp.npy)
        # "hidden_size" è il numero di unità nascoste nel LSTM (inluenza la capacità del modello di memorizzare, più è alto il valore più precisione si ottiene ma cè un maggiore rischio di overfitting)
        # "num_classes" è il numero di classi da predire (es. 24 nel caso delle AU)
          
        # Layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)  # Layer che è in grado di imparare informazioni su sequenze temporali
        # batch_first = True significa che l'input e l'output avranno la stessa forma, ovvero (batch_size, sequence_length, input_size)

        # Layer Fully Connected (FC)
        self.fc1 = nn.Linear(hidden_size, 128)  # Viene ridotta la dimensione dell'output del LSTM a 128
        self.fc2 = nn.Linear(128, 64)  # Viene ridotta la dimensione dell'output del FC1 a 64
        self.fc3 = nn.Linear(64, num_classes)  # Viene ridotta la dimensione dell'output del FC2 al numero di classi da predire

        self.relu = nn.ReLU()  # Funzione di attivazione ReLU
        #self.softmax = nn.Softmax(dim = 1)  # Funzione softmax per la classificazione multi-classe
        self.dropout = nn.Dropout(0.5)  # Dropout per evitare l'overfitting (durante il training, a ogni passo, vengono disattivate alcune unità in modo casuale)

    def forward(self, x):
        # Si ipotizza che l'input sia di forma: batch_size (dimensione del batch), sequence_length, input_size
        # Forward pass attraverso il layer LSTM
        print("Input to LSTM", x.shape)
        lstm_out, (hn, cn) = self.lstm(x)
        #print("LSTM output shape", lstm_out.shape)
        lstm_out = lstm_out

        # Forward pass attraverso il primo layer FC
        x = self.fc1(lstm_out)
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

    train_indices = torch.arange(unique_subject_indices[0], unique_subject_indices[num_train_subjects])
    test_indices = torch.arange(unique_subject_indices[num_train_subjects], unique_subject_indices[-1])

    train_data = data[train_indices]
    test_data = data[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    return train_data, test_data, train_labels, test_labels

def main():
    root_dir = '/andromeda/personal/fgastaldi/facemorphic_output'  # CAMBIARE CON LA DIRECTORY CORRETTA DEL DATASET DI RIFERIMENTO
    n_frames = 40  # Numero di frame da prelevare per ciascun video
    n_features = 106  # Numero di features per frame (es. dimensione di exp.npy)
    # globalpose.npy contiene 3 elementi, jawpose.npy contiene 3 elementi, exp.npy contiene 100 elementi, shape.npy contiene 300 elementi
    # In funzione di quali file si vogliono utilizzare per il training/test il numero di features deve essere corretto!
    
    max_seq_length = n_frames*n_features  # Imposta la lunghezza massima delle sequenze!

    dataset = FacialExpressionDataset(root_dir, max_seq_length = max_seq_length) # Viene creato il dataset
    #print(dataset.get_processed_subjects()) # Viene stampato l'ordine con cui i soggetti sono stati processati

    train_data, test_data, train_labels, test_labels = split_train_test_data(dataset.data, dataset.labels)

    '''
    print(len(dataset.data))
    print(train_data)
    print(train_data.shape)
    print(test_data)
    print(test_data.shape)
    print(train_labels)
    print(train_labels.shape)
    print(test_labels)
    print(test_labels.shape)
    '''

    train_labels = train_labels.clone().detach()
    test_labels = test_labels.clone().detach()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    input_size = dataset.data[0].shape[0]  # Numero di features per video, ovvero 40 (numero di frames) x 100 (numero di features per frame nel caso di exp.npy)
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
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    plt.savefig('confusion_matrixV2.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()

