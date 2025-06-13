import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

'''
/andromeda/personal/fgastaldi/facemorphic_output/
├── 7ee789dd670142a19522f75b54a96eb7/
│   ├── AU_1/
│   │   ├── frames_1/
│   │   │   ├── frame_001/
│   │   │   │   └── exp.npy
│   │   │   ├── frame_002/
│   │   │   │   └── exp.npy
│   │   │   └── ...
│   │   ├── frames_2/
│   │   │   ├── frame_001/
│   │   │   │   └── exp.npy
│   │   │   └── ...
│   │   └── ...
│   ├── AU_2/
│   │   ├── frames_1/
│   │   │   ├── frame_001/
│   │   │   │   └── exp.npy
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── altro_soggetto/
'''

class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, subject = None, max_seq_length = None):
        # root_dir: directory contenente i dati da caricare
        # subject: soggetto da considerare (se non specificato, vengono considerati tutti i soggetti)
        # max_seq_length: lunghezza massima delle sequenze (se non specificato, le sequenze non vengono tagliate)

        self.data = []  # Lista per contenere le sequenze dei valori contenuti nei file exp.npy
        self.labels = []  # Lista per contenere le etichette corrispondenti alle sequenze
        self.label_map = {}  # Dizionario per mappare le etichette (AU_1, AU_2...) in interi
        
        # Contatori per verificare il numero di frames che vengono "paddati" e che non vengono
        self.padding_counter = 0
        self.no_padding_counter = 0

        self.subjects = []  # Lista dei soggetti da considerare
        if subject:
            self.subjects = [subject]
        else:
            # Se non è stato specificato un soggetto, considera tutti i soggetti presenti nella directory
            self.subjects = os.listdir(root_dir)
        
        self.max_seq_length = max_seq_length

        # CICLO PER OGNI SOGGETTO (CARTELLE CON NOMI RANDOM)
        for subject in self.subjects:
            subject_path = os.path.join(root_dir, subject)
            #print(subject_path)
            if not os.path.isdir(subject_path):
                raise ValueError(f"Il soggetto {subject} non esiste in {root_dir}")
            
            # CICLO PER OGNI ACTION UNIT (AU_1, AU_2...) DEL SOGGETTO IN ANALISI
            for au_folder in os.listdir(subject_path):
                au_path = os.path.join(subject_path, au_folder)
                #print(au_path)
                if not os.path.isdir(au_path):
                    continue
                
                # CICLO PER OGNI CARTELLA "FRAMES" DELL'ACTION UNIT
                for frames_folder in os.listdir(au_path):
                    frames_path = os.path.join(au_path, frames_folder)
                    #print(frames_path)
                    if not os.path.isdir(frames_path):
                        continue
                    # A questo punto, ad esempio, si è in: /7ee789dd670142a19522f75b54a96eb7/AU_1/frames_1
                    
                    frames = []  # Lista vuota per memorizzare i frame (sequenze temporali)
                    # CICLO PER OGNI FRAME CONTENUTO NELLA CARTELLA "FRAMES" DELL'ACTION UNIT
                    for frame_folder in sorted(os.listdir(frames_path)):
                        frame_path = os.path.join(frames_path, frame_folder, "exp.npy")  # CAMBIARE CON IL NOME DEL FILE CORRETTO (cam.npy, exp.npy, globalpose.npy, jawpose.npy, lightcode.npy, shape.npy, tex.npy)
                        if os.path.exists(frame_path):
                            exp_data = np.load(frame_path)  # Viene caricato il file exp.npy corrispondente al frame in analisi
                            frames.append(torch.tensor(exp_data, dtype = torch.float32))  # Il file exp.npy viene convertito in un tensore PyTorch aggiunto alla lista dei frame  
                    #print(frames)
                    #print(len(frames))

                    if len(frames) > 0: # La lunghezza di frames sarà pasi al numero di frame contenuti nella cartella "frames" (ovvero la lunghezza del video) 
                        if self.max_seq_length: # Se "max_seq_length" è specificato
                            if len(frames) < self.max_seq_length: # Viene fatto il padding fino alla lunghezza massima andando ad aggiungere tensori di zeri fino a raggiungere la lunghezza massima
                                frames.extend([torch.zeros_like(frames[0])] * (self.max_seq_length - len(frames)))
                                self.padding_counter += 1
                                #print(frames)
                                #print(len(frames))
                            else: # Se invece il numero dei frames è maggiore della lunghezza massima allora vengono troncati i frame in eccesso
                                frames = frames[-self.max_seq_length:]
                                self.no_padding_counter += 1
                                #print(frames)
                                #print(len(frames))
                        # Aggiungi la sequenza di frames al dataset
                        self.data.append(torch.stack(frames))
                        #print(len(self.data))
                        #print(self.data[0].shape)
                        # Le righe di codice sottostanti servono per mappare le etichette (AU_1, AU_2...) in interi
                        if au_folder not in self.label_map:
                            self.label_map[au_folder] = len(self.label_map)
                        self.labels.append(self.label_map[au_folder])

        # Alla fine del ciclo self.data è una lista di tensori PyTorch, ognuno dei quali ha dimensioni: numero di frame (10) x numero di features per frame (100 nel caso di exp.npy)
        # La lunghezza della lista sarà pari al numero totale di video presenti nella directory.
        
        self.labels = torch.tensor(self.labels, dtype = torch.long)  # La lista delle etichette viene convertita in un tensore PyTorch
        self.reverse_label_map = {v: k for k, v in self.label_map.items()} # "Mappatura" intero che rappresenta l'etichetta con l'action unit corrispondente (es. 0 : 'AU_53')
        #print(self.reverse_label_map)

        print(f"Numero di frame paddati: {self.padding_counter}")
        print(f"Numero di frame non paddati: {self.no_padding_counter}")

    def __len__(self):
        return len(self.data)  # Restituisce la lunghezza del dataset

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Restituisce i dati e le etichette per un dato indice


class FacialExpressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FacialExpressionModel, self).__init__()
        # "input_size" è il numero di features per frame (es. 100 nel caso dei file exp.npy)
        # "hidden_size" è il numero di unità nascoste nel LSTM (inluenza la capacità del modello di memorizzare)
        # "num_classes" è il numero di classi da predire (es. 24 nel caso delle action unit)
          
        #LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)  # In questo modo vengono processate sequenze temporali
        # Layer Fully Connected (FC)
        self.fc1 = nn.Linear(hidden_size, 128)  # Viene ridotta la dimensione dell'output del LSTM a 128
        self.fc2 = nn.Linear(128, 64)  # Viene ridotta la dimensione dell'output del FC1 a 64
        self.fc3 = nn.Linear(64, num_classes)  # Viene ridotta la dimensione dell'output del FC2 al numero di classi da predire

        self.relu = nn.ReLU()  # Funzione di attivazione ReLU
        self.softmax = nn.Softmax(dim = 1)  # Funzione softmax per la classificazione multi-classe
        self.dropout = nn.Dropout(0.5)  # Dropout per evitare l'overfitting (durante il training, a ogni passo, vengono disattivate alcune unità in modo casuale)

    def forward(self, x):
        # Si ipotizza che l'input sia di forma: batch_size (dimensione del batch), sequence_length, input_size (100 nel caso si valutino i file exp.npy)
        # Forward pass attraverso il layer LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = hn[-1]
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
        return self.softmax(x)  # Restituisce l'output della rete


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


def main():
    root_dir = '/andromeda/personal/fgastaldi/facemorphic_output'  # CAMBIARE CON LA DIRECTORY CORRETTA DEL DATASET DI RIFERIMENTO
    max_seq_length = 20  # Imposta la lunghezza massima delle sequenze! Sarebbe il numero massimo di frame per ciascun video
    # L'idea è quella di prendere gli ultimi 10 frame di ciascuna video.
    # In particolare, se un video ha meno di 10 frame, allora viene effettuato il padding, altrimenti si prendono gli ultimi 10!

    dataset = FacialExpressionDataset(root_dir, max_seq_length = max_seq_length)

    # Funzione di scikit-learn per dividere il dataset in training e test set. Il test set è il 20% del dataset totale.
    train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.labels, test_size = 0.2, random_state = 42)

    train_data = torch.stack(train_data)
    test_data = torch.stack(test_data)
    train_labels = train_labels.clone().detach()
    test_labels = test_labels.clone().detach()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    input_size = dataset.data[0].shape[1]  # Numero di features per frame (es. exp.npy dimension)
    hidden_size = 64  # Numero di unità nel LSTM
    num_classes = len(dataset.label_map)

    model = FacialExpressionModel(input_size, hidden_size, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
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
    plt.savefig('confusion_matrix.png', bbox_inches='tight')


if __name__ == "__main__":
    main()

