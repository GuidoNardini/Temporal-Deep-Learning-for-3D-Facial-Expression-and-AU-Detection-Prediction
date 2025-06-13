import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, subject=None):
        self.data = []
        self.labels = []
        self.label_map = {}
        
        self.subjects = []
        if subject:
            self.subjects = [subject]
        else:
            self.subjects = os.listdir(root_dir)
        
        for subject in self.subjects:
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                raise ValueError(f"Il soggetto {subject} non esiste in {root_dir}")

            for au_folder in os.listdir(subject_path):
                au_path = os.path.join(subject_path, au_folder)
                if not os.path.isdir(au_path):
                    continue

                for frames_folder in os.listdir(au_path):
                    frames_path = os.path.join(au_path, frames_folder)
                    if not os.path.isdir(frames_path):
                        continue

                    last_frame_folder = sorted(os.listdir(frames_path))[-1]
                    last_frame_path = os.path.join(frames_path, last_frame_folder, "exp.npy") # CAMBIARE CON IL NOME DEL FILE CORRETTO

                    if os.path.exists(last_frame_path):
                        exp_data = np.load(last_frame_path)
                        self.data.append(exp_data)

                        if au_folder not in self.label_map:
                            self.label_map[au_folder] = len(self.label_map)
                        self.labels.append(self.label_map[au_folder])

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class FacialExpressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FacialExpressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)
    
class EarlyStopping:
    def __init__(self, patience = 5, delta = 0.0):
        self.patience = patience
        self.delta = delta
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = '/andromeda/personal/fgastaldi/facemorphic_output'  # CAMBIARE CON LA DIRECTORY CORRETTA
    dataset = FacialExpressionDataset(root_dir)
    #print(f"Dataset size: {len(dataset)}")
    #print(f"Labels: {dataset.labels}")
    #print(f'Data shape: {dataset.data.shape}')

    train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.labels, test_size = 0.2, random_state = 42)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    input_size = dataset.data.shape[1]
    num_classes = len(dataset.label_map)

    model = FacialExpressionModel(input_size, num_classes).to(device)
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
        
        scheduler.step(running_loss/len(train_loader))

        if early_stopping(running_loss/len(train_loader)):
            print("Early stopping")
            break
  
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    print("Training complete!")


    # TESTING
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy} %")

if __name__ == '__main__':
    main()



