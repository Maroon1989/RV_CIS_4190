import numpy as np
import pandas as pd
def wap1(df:pd.DataFrame) -> pd.Series:
    def calc_wap1(df: pd.DataFrame) -> pd.Series:
        wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
        return wap
    df['wap1'] = calc_wap1(df)
    return df

def wap2(df:pd.DataFrame) -> pd.Series:
    def calc_wap2(df: pd.DataFrame) -> pd.Series:
        wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
        return wap
    df['wap2'] = calc_wap2(df)
    return df

