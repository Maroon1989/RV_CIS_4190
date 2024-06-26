import sys
sys.path.append('Model')
sys.path.append('Factor')
import numpy as np
import pandas as pd
import os
from joblib import Parallel,delayed
from config import BOOK_PATH,TRADE_PATH,MEMORY_MODE
import traceback
from contextlib import contextmanager
import time
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Model import STRATEGIES,DEEP
from Factor import FACTORS
@contextmanager
def timer(name:str):
    s = time.time()
    yield
    elapsed = time.time()-s
    print(f'[{name}] {elapsed: .3f}sec')
    
def print_trace(name: str = ''):
    print(f'ERROR RAISED IN {name or "anonymous"}')
    print(traceback.format_exc())

def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ['time_id', 'stock_id']:
            ret.append(c[0])
        else:
            ret.append('.'.join([prefix] + list(c)))
    return ret

class Factor:
    def __init__(self) -> None:
        self.book_path = BOOK_PATH
        self.trade_path = TRADE_PATH
        self.stock_ids = set(os.listdir(self.book_path))
        self.book_dst = {}
        self.trade_dst = {}
        self.trade = None
        self.book = None
        self.data = None
        self.factor_dict = FACTORS

    def load_data(self,stock_id,book):
        # for i in stock_ids:
        #     file_path = 
        #     data = pd.read_csv(self.path)
        self.path = self.book_path if book else self.trade_path
        dir_path = os.path.join(self.path,stock_id)
        file = os.listdir(dir_path)[0]
        file_path = os.path.join(dir_path,file)
        data = pd.read_parquet(file_path)
        data['stock_id'] = stock_id
        return data
    
    def load_all_data(self,book):
        if book:
            with timer('load_book_data'):
                book = Parallel(n_jobs=-1)(delayed(self.load_data)(stock_id,True) for stock_id in self.stock_ids)
                self.book = pd.concat(book)
                self.book = self.book.reset_index(drop=True)
        else:
            with timer('load_trade_data'):
                trade = Parallel(n_jobs=-1)(delayed(self.load_data)(stock_id,False) for stock_id in self.stock_ids)
                self.trade = pd.concat(trade)
                self.trade = self.trade.reset_index(drop=True)

    def book_feature(self,func_call,stock_id):
        # func_call = {'name':[func,[np.mean,np.sum,etc.]]}
        data = self.load_data(stock_id,True)
        print(1)
        name = list(func_call.keys())[0]
        func_list = func_call[name][1]
        df = func_call[name][0](data)
        df['stock_id'] = stock_id
        if func_list == None:
            func_list = [np.mean]
        self.book_dst[name] = func_list
        # print(df)
        # logging.INFO(df)
        return df
    
    def book_feature2(self,func_call,stock_id):
        data = self.book[self.book['stock_id']==stock_id]
        name = list(func_call.keys())[0]
        func_list = func_call[name][1]
        df = func_call[name][0](data)
        df['stock_id'] = stock_id
        if func_list == None:
            func_list = [np.mean]
        self.book_dst[name] = func_list
        # print(df)
        # logging.INFO(df)
        return df
    
    def trade_feature(self,func_call,stock_id):
        # func_call = {'name':[func,[np.mean,np.sum,etc.]]}
        data = self.load_data(stock_id,False).copy()
        name = list(func_call.keys())[0]
        func_list = func_call[name][1]
        df = func_call[name][0](data)
        df['stock_id'] = stock_id
        if func_list == None:
            func_list = [np.mean]
        self.trade_dst[name] = func_list
        # print(df)
        return df
    
    def trade_feature2(self,func_call,stock_id):
        data = self.trade[self.trade['stock_id']==stock_id]
        name = list(func_call.keys())[0]
        func_list = func_call[name][1]
        df = func_call[name][0](data)
        df['stock_id'] = stock_id
        if func_list == None:
            func_list = [np.mean]
        self.trade_dst[name] = func_list
        # print(df)
        return df
    
    def make_trade_feature(self,func_call):
        with timer('trade'):
            trades = Parallel(n_jobs=-1)(delayed(self.trade_feature)(func_call,stock_id) for stock_id in self.stock_ids)
            self.trade = pd.concat(trades)
        return self.trade
    
    def make_book_feature(self,func_call):
        with timer('books'):
            books = Parallel(n_jobs=-1)(delayed(self.book_feature)(func_call,stock_id) for stock_id in self.stock_ids)
            self.book = pd.concat(books)
        return self.book
    
    def assemble_book_feature(self,book):
        if book:
            for key,value in tqdm(self.factor_dict.items()):
                func_call = {key:value}
                books = Parallel(n_jobs=-1)(delayed(self.book_feature2)(func_call,stock_id) for stock_id in self.stock_ids)
                self.book = pd.concat(books)
                self.book = self.book.reset_index(drop=True)
        else:
            for key,value in tqdm(self.factor_dict.items()):
                func_call = {key:value}
                books = Parallel(n_jobs=-1)(delayed(self.trade_feature2)(func_call,stock_id) for stock_id in self.stock_ids)
                self.trade = pd.concat(books)
                self.trade = self.book.reset_index(drop=True)
            
    def make_features(self,book,stock_id):
        dst = self.book_dst if book else self.trade_dst
        # data = self.load_data(stock_id,book).copy()
        # data = self.book.copy() if book else self.trade.copy()
        data = self.book[self.book['stock_id']==stock_id].reset_index(drop=True) if book else self.trade[self.trade['stock_id']==stock_id].reset_index(drop=True)
        agg = data.groupby('time_id').agg(dst).reset_index()
        if book:
            agg.columns = flatten_name('book',agg.columns)
            agg['stock_id'] = stock_id
            for time in tqdm([450,300,150]):
                d = data[data['seconds_in_bucket'] >= time].groupby('time_id').agg(dst).reset_index(drop=False)
                d.columns = flatten_name(f'book_{time}', d.columns)
                agg = pd.merge(agg, d, on='time_id', how='left')
            return agg
        else:
            agg.columns = flatten_name('trade',agg.columns)
            agg['stock_id'] = stock_id
            for time in tqdm([450,300,150]):
                d = data[data['seconds_in_bucket'] >= time].groupby('time_id').agg(dst).reset_index(drop=False)
                d.columns = flatten_name(f'trade_{time}', d.columns)
                agg = pd.merge(agg, d, on='time_id', how='left')
            return agg
    def assemble(self,base):
        with timer('books'):
            book = Parallel(n_jobs=10,verbose=2)(delayed(self.make_features)(True,stock_id) for stock_id in self.stock_ids)
            books = pd.concat(book)
            print('books assemble successfully!')
            print(book)
        try:
            with timer('trade'):
                trade = Parallel(n_jobs=10,verbose=2)(delayed(self.make_features)(False,stock_id) for stock_id in self.stock_ids)
                trades = pd.concat(trade)
                print('trades assemble successfully!')
        except:
            trades = pd.DataFrame(columns=['stock_id','time_id'])
        with timer('assemble all'):
            if MEMORY_MODE:
                df = pd.merge(base,books,on=['stock_id', 'time_id'], how='left')
                df = pd.merge(df,trades,on=['stock_id', 'time_id'], how='left')
            else:
                df = pd.merge(books,trades,on=['stock_id', 'time_id'], how='left')
        self.data = df
        return df
    def make_model(self,func,X_train,y_train,X_test,y_test):
        y_pred = func(X_train,y_train,X_test,y_test)
        return y_pred
    def essemble_model(self):
        # X_train,y
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        # y_test, y_pred  = value(X,y)
        with timer('Machine Learning Models'):
            y_pred = Parallel(n_jobs=-1,verbose=2)(delayed(self.make_model)(value,X_train,X_test,y_train,y_test) for key,value in STRATEGIES.items())
            y_preds = pd.concat(y_pred,axis=1)
        with timer('Deep Learning Models'):
            # y_pred = Parallel(n_jobs=-1,verbose=2)(delayed(self.make_model)(value) for key,value in DEEP.items())
            # y_preds = pd.concat(y_pred,axis=1)
            for key, value in DEEP.items():
                y_pred = self.make_model(value,X_train,X_test,y_train,y_test)
                y_preds = pd.concat([y_preds,y_pred])
        predictions = y_preds*self.weights
        return predictions
    
    # print(1)
    
        # if book:
        #     with timer('book'):
    # def make_feature(self,func_call,stock_id,book):
    #     # func_call = {'name':[func,[np.mean,np.sum,etc.]]}
    #     data = self.load_data(stock_id,book).copy()
    #     name = func_call.keys()[0]
    #     func_list = func_call[name][1]
    #     data = func_call[name][0](data)
    #     dst = self.book_dst if book else self.trade_dst
    #     dst[name] = func_list
        # return name,func_list
        # data.to
    # def add_funcs(self,data):
    #     # data = self.load_data(stock_id,book).copy()
    #     # dst = func_call(data)
    #     func_list = dst
    #     # factor = func_call(data)
    #     return data
    

        # pass
