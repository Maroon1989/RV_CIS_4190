import sys
sys.path.append(r'D:\Upenn\CIS 4190\project\RV_CIS_4190\Model')
from Model.model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
data = pd.read_csv('data.csv')
data = data.dropna()
X = data.drop(columns=['time_id','stock_id','RV'])
y = data['RV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# data = pd.read_csv('data.csv')
y_pred = build_CNN(X_train,y_train,X_test,y_test)
# print(y_pred)
# data = data[:min(len(data),len(y_pred))]
results = pd.DataFrame()
results['CNN'] = y_pred
y_pred = build_nn(X_train,y_train,X_test,y_test)
results['NN'] = y_pred
y_pred = build_xgboost(X_train,y_train,X_test,y_test)
results['XGBoost'] = y_pred
y_pred = build_lasso(X_train,y_train,X_test,y_test)
results['Lasso'] = y_pred
y_pred = build_Ridge(X_train,y_train,X_test,y_test)
results['Ridge'] = y_pred
results['target'] = y_test
results[['CNN','NN','XGBoost','Lasso','Ridge','target']].to_csv('brunch_results.csv',index=False)