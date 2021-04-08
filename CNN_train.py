from sklearn.metrics import confusion_matrix, classification_report

import CNN_net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *

EPOCHS = 28
BATCH_SIZE = 20
LEARNING_RATE = 0.001

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = LEARNING_RATE * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LoadFirstDataset(Dataset):
    def __init__(self, path):
        data, y_x, y_y = load_dataset(path, flatten=False)

        data = np.array(data)
        data = data/255
        print(data.shape)
        data = torch.from_numpy(data)
        print(data.shape)
        data = data.reshape(len(data), 1, 64, 64)

        self.X = data
        print("size:", self.X.shape)

        y_x = np.array(y_x).reshape((len(y_x), 1))
        y_y = np.array(y_y).reshape((len(y_y), 1))
        Y = []
        for x, y in zip(y_x, y_y):
            Y.append(point2grid(x, y, 4, 4))

        self.y = torch.from_numpy(np.array(Y))
        self.y = self.y.type(torch.LongTensor)
        print(type(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

def prepare_data(path):
    dataset = LoadFirstDataset(path)
    train, test = random_split(dataset, [800, 309])
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=True)
    return train_dl, test_dl

def train_model(train_loader, model):
    global LEARNING_RATE
    model.train()

    #criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        count = 0

        for X_batch, y_batch in train_loader:
            #adjust_learning_rate(optimizer, e)

            print(count, end=' ')
            count += 1

            #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            X_batch = X_batch.float()
            y_pred = model(X_batch)

            #y_batch = y_batch.float()

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.numpy()

        print("")
        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f}')

def evaluate_model(test_dl, model):
    pred_list = []
    true_list = []
    model.eval()
    first = True
    with torch.no_grad():

        for X_batch, Y_batch in test_dl:



            true_list.append(Y_batch)

            X_batch = X_batch.to(device)
            X_batch = X_batch.float()
            pred = model(X_batch)

            max_value, predicted = torch.max(pred, 1)
            pred_list.append(predicted)

            if(first):
                print("pred list",pred_list)
                print("true list",true_list)

                print("ybatch",Y_batch)
                print("y pred tag", pred)
                print("predicted", predicted)
                print("max_value", max_value)

                first = False


    y_pred = [a.squeeze().tolist() for a in pred_list]
    y_test = [a.squeeze().tolist() for a in true_list]

    print(y_pred)
    print(y_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = "pictures/stable2"
    save_path = "models/CNN5_STABLE2.pt"

    # prepare data
    print("preparing the data")
    train_dl, test_dl = prepare_data(dataset_path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # train the model
    model = CNN_net.CNN5(1)
    model = model.float()
    print("train the model")
    train_model(train_dl, model)
    #model.load_state_dict(torch.load(save_path))
    #model.eval()

    #saving the model
    print("saving the model")
    torch.save(model.state_dict(), save_path)

    #evaluating the model
    print("evaluating the model")
    evaluate_model(test_dl, model)

