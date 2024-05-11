from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import torch.optim as optim
from  torch.utils.data import DataLoader,TensorDataset
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
    y_pred = pd.DataFrame({'Lasso_pred':y_pred})
    return y_pred
def build_xgboost(X_train,y_train,X_test,y_test):
    model = model = xgb.XGBRegressor(n_estimators=1000,max_depth=10)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame({'Xgboost_pred':y_pred})
    return y_pred

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 1)  # Assuming a single output for regression

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_CNN(x_train, y_train, x_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = CNN()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueezing to match output shape
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Evaluate the model
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.squeeze().tolist())
    y_pred = pd.DataFrame({'CNN_pred':y_pred})
    return y_pred