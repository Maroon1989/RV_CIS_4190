from sklearn.linear_model import Lasso,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import torch.optim as optim
from  torch.utils.data import DataLoader,TensorDataset,Dataset
from tqdm import tqdm
def build_lasso(X_train,y_train,X_test,y_test):
    # split dataset
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = Lasso(alpha=0.01)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # model.fit(X_train,y_train)                          # cross validation     
    model.fit(X_train,y_train)                            # output test_pred      tune para                                                                                                                                                                                                                                                                                                                                          
    y_pred = model.predict(X_test)
    # y_pred = pd.DataFrame({'Lasso_pred':y_pred})
    return y_pred

def build_Ridge(X_train,y_train,X_test,y_test):
    # split dataset
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = Ridge(alpha=0.01)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # model.fit(X_train,y_train)                          # cross validation     
    model.fit(X_train,y_train)                            # output test_pred      tune para                                                                                                                                                                                                                                                                                                                                          
    y_pred = model.predict(X_test)
    # y_pred = pd.DataFrame({'Lasso_pred':y_pred})
    return y_pred

def build_xgboost(X_train,y_train,X_test,y_test):
    model = model = xgb.XGBRegressor(n_estimators=1000,max_depth=10)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    # y_pred = pd.DataFrame({'Xgboost_pred':y_pred})
    return y_pred

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # STUDENT TODO START: Create the layers of your CNN here
        # Convolutional layers
        self.conv1 = nn.Conv1d(32, 16, kernel_size=1)
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=1)
        # Fully connected layers
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)
        # STUDENT TODO END

    def forward(self, x):
        # STUDENT TODO START: Perform the forward pass through the layers
        # print(x.shape,1)
        x = self.pool(torch.relu(self.conv1(x)))
        # print(x.shape,2)
        x = x.view(-1,16)
        # print(x.shape,3)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        # print(x.shape,4)
        x = self.fc2(x)
        # print(x.shape,5)
        # x = x.view(-1,1)
        return x
        
    #     super(CNN, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=2,padding=1)
    #     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2,padding=1)
    #     self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2,padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.fc1 = nn.Linear(4 , 64)
    #     self.fc2 = nn.Linear(64, 1)  # Assuming a single output for regression

    # def forward(self, x):
    #     x = self.pool(torch.relu(self.conv1(x)))
    #     x = self.pool(torch.relu(self.conv2(x)))
    #     x = self.pool(torch.relu(self.conv3(x)))
    #     x = torch.flatten(x, 1)
    #     x = torch.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    
# class CNN_dataset(Dataset):
#     def __init__(self,x,y):


def build_CNN(x_train, y_train, x_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x_train = torch.tensor(x_train.to_numpy(),dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32)
    x_test = torch.tensor(x_test.to_numpy(),dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32)

    x_train = x_train.unsqueeze(1).transpose(2,1)
    x_test = x_test.unsqueeze(1).transpose(2,1)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = CNN()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(10)):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.shape)
            # print(labels.shape)
            # print(labels.unsqueeze(1).shape)
            outputs = model(inputs)
            # print(outputs.shape)
            # print(labels.shape,labels.unsqueeze(1).shape)
            loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueezing to match output shape
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Evaluate the model
    model.eval()
    y_pred = []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            # print(outputs)
            # print(outputs.shape)
            # y_pred.append(outputs.squeeze().item())
            test_loss+=loss.item()
            y_pred.extend(outputs.view(-1).cpu().numpy())
        print(f"Test Loss: {test_loss/len(test_loader)}")   
    # y_pred = pd.DataFrame({'CNN_pred':y_pred})
    return y_pred

import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络用于回归
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)  # 输出层只有一个神经元，输出连续值
        self.fc3 = nn.Linear(64,1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def build_nn(X_train, y_train, X_test, y_test, epochs=5, learning_rate=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转换数据类型
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)

    input_size = X_train.shape[1]
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = SimpleNN(input_size, 100).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.shape)
            # print(labels.shape)
            # print(labels.unsqueeze(1).shape)
            outputs = model(inputs)
            # print(outputs.shape)
            # print(labels.shape,labels.unsqueeze(1).shape)
            loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueezing to match output shape
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    y_pred = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            # print(outputs)
            # print(outputs.shape)
            # y_pred.append(outputs.squeeze().item())
            test_loss+=loss.item()
            y_pred.extend(outputs.view(-1).cpu().numpy())
        print(f"Test Loss: {test_loss/len(test_loader)}") 
        # y_pred.extend(predicted)
    # y_pred = pd.DataFrame({'NN_pred':y_pred})
    return y_pred

# 使用示例：
# y_pred, test_loss = nn_train_and_evaluate(X_train, y_train, X_test, y_test)
# print(f'Test Loss: {test_loss}')
# print(f'Predicted values: {y_pred}')
