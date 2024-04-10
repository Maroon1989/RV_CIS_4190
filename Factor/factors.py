import numpy as np
import pandas as pd
def wap1(df:pd.DataFrame) -> pd.Series:
    def calc_wap1(df: pd.DataFrame) -> pd.Series:
        wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
        return wap
    df['wap1'] = calc_wap1(df)
    return df