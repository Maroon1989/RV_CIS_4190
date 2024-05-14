from sklearn.metrics import mean_squared_error
from loss_func import rmspe
import pandas as pd
import numpy as np
results = pd.read_csv('brunch_results.csv')
results['target'] = results['target'].fillna(results['target'].mean())
for column in results.columns:
    print(column)
    if column=='target':
        break
    y_pred = results[column]
    y_true = results['target']
    # y_true = np.fillna()
    print(f'{column}: mse: {mean_squared_error(y_true,y_pred)}, rmspe: {rmspe(y_true,y_pred)}')
w= [1/5,9/25,3/25,7/25,1/25]
y_pred = results[['CNN','NN','XGBoost','Lasso','Ridge']].mean(axis=1).tolist()
y_pred = results['CNN']*w[0]+results['NN']*w[1]+results['XGBoost']*w[2]+results['Lasso']*w[3]+results['Ridge']*w[4]
# print(y_pred)
print(f'final: mse: {mean_squared_error(y_true,y_pred)}, rmspe: {rmspe(y_true,y_pred)}')