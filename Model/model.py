from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
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
