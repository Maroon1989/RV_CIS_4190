import numpy as np
import pandas as pd
import scipy.stats as stats

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
# 计算feature的函数,贴个注释
def wap1(df:pd.DataFrame) -> pd.Series:
    def calc_wap1(df: pd.DataFrame) -> pd.Series:
        wap = (df['bid_price1'] * df['ask_price1'] + df['ask_price1'] * df['bid_price1']) / (df['bid_price1'] + df['ask_price1'])
        return wap
    df['wap1'] = calc_wap1(df)
    return df

def calc_wap(df: pd.DataFrame) -> pd.DataFrame:
    df["wap1"] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return df

def rtn5_mean(df: pd.DataFrame) -> pd.DataFrame:
    def cal_rtn5_mean(df: pd.DataFrame) -> pd.DataFrame:
        """
        收益率 5 期均值
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        rtn = np.nanmean(p[5:] / p[:-5] - 1)
        df["rtn5_mean"] = rtn
        return df
    df = df.groupby("time_id").apply(cal_rtn5_mean).reset_index(drop=True)
    return df

def real_var(df: pd.DataFrame) -> pd.DataFrame:
    def cal_real_var(df: pd.DataFrame) -> pd.DataFrame:
        """
        已实现波动率
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        rtn = np.log(p[1:] / p[:-1])
        var = np.sqrt(np.sum(rtn ** 2))
        df["real_var"] = var
        return df
    df = df.groupby("time_id").apply(cal_real_var).reset_index(drop=True)
    return df

def real_skew(df: pd.DataFrame) -> pd.DataFrame:
    def cal_real_skew(df: pd.DataFrame) -> pd.DataFrame:
        """
        收益率偏度
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        rtn = np.log(p[1:] / p[:-1])
        skew = np.sum(rtn ** 3) / (np.sum(rtn ** 2) ** 1.5)
        df["real_skew"] = skew
        return df
    df = df.groupby("time_id").apply(cal_real_skew).reset_index(drop=True)
    return df

def real_kurt(df: pd.DataFrame) -> pd.DataFrame:
    def cal_real_kurt(df: pd.DataFrame) -> pd.DataFrame:
        """
        收益率峰度
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        rtn = np.log(p[1:] / p[:-1])
        kurt = np.sum(rtn ** 4) / (np.sum(rtn ** 2) ** 2)
        df["real_kurt"] = kurt
        return df
    df = df.groupby("time_id").apply(cal_real_kurt).reset_index(drop=True)
    return df

def rv_up(df: pd.DataFrame) -> pd.DataFrame:
    def cal_rv_up(df: pd.DataFrame) -> pd.DataFrame:
        """
        上行收益率已实现方差
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        rtn = np.log(p[1:] / p[:-1])
        var_up = np.nansum(rtn[rtn > 0] ** 2)
        df["rv_up"] = var_up
        return df
    df = df.groupby("time_id").apply(cal_rv_up).reset_index(drop=True)
    return df

def rv_down(df: pd.DataFrame) -> pd.DataFrame:
    def cal_rv_down(df: pd.DataFrame) -> pd.DataFrame:
        """
        下行收益率已实现方差
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        rtn = np.log(p[1:] / p[:-1])
        var_down = np.nansum(rtn[rtn < 0] ** 2)
        df["rv_down"] = var_down
        return df
    df = df.groupby("time_id").apply(cal_rv_down).reset_index(drop=True)
    return df

def nog_gs(df: pd.DataFrame) -> pd.DataFrame:
    def cal_nog_gs(df: pd.DataFrame) -> pd.DataFrame:
        """
        收益率噪音偏离：基于高斯核密度估计收益率偏离正态分布的程度
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        
        p_now = p[10:]
        p_pre = p[:-10]
        rtn = np.log(p_now / p_pre)
        rtn_mean = np.nanmean(rtn)
        rtn_std = np.nanstd(rtn)
        rtn_nor = np.sort((rtn - rtn_mean) / rtn_std)
        rtn_nor_min = np.min(rtn_nor)
        rtn_nor_max = np.max(rtn_nor)
        rtn_nor_diff = rtn_nor_max - rtn_nor_min
        p = len(rtn_nor)
        if p > 10 and (not np.isnan(rtn_nor_diff)):
            q = stats.norm.pdf(np.arange(rtn_nor_min, rtn_nor_max, rtn_nor_diff / p))

            scipy_kde = stats.gaussian_kde(rtn_nor)
            dens = scipy_kde.pdf(rtn_nor)
            wht_nos = dens - q[0:p]
        elif np.isnan(rtn_nor_diff):
            wht_nos = rtn_nor
        else:
            wht_nos = np.nan
        df["nog_gs"] = np.nansum(wht_nos ** 2)
        return df
    df = df.groupby("time_id").apply(cal_nog_gs).reset_index(drop=True)
    return df

def exRtn_maxVal(df: pd.DataFrame) -> pd.DataFrame:
    def cal_exRtn_maxVal(df: pd.DataFrame) -> pd.DataFrame:
        """
        收益率极大值幅度：基于 VaR 计算收益率的极大值均值
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        p_now = p[5:]
        p_pre = p[:-5]
        rtn = np.log(p_now / p_pre)
        rtn_mean = np.nanmean(rtn)
        rtn_std = np.nanstd(rtn)
        Extre_num = rtn_mean + 1.96 * rtn_std
        Extre_fre = rtn - Extre_num
        Extre_rel = Extre_fre[Extre_fre > 0] / Extre_num
        df["exRtn_maxVal"] = np.nansum(Extre_rel)
        return df
    df = df.groupby("time_id").apply(cal_exRtn_maxVal).reset_index(drop=True)
    return df

def exRtn_minFre(df: pd.DataFrame) -> pd.DataFrame:
    def cal_exRtn_minFre(df: pd.DataFrame) -> pd.DataFrame:
        """
        收益率极小值频率：基于 VaR 计算收益率的极小值频率
        """
        try:
            p = df["wap1"].values
        except:
            df = calc_wap(df)
            p = df["wap1"].values
        p = p[p > 0]
        p_now = p[5:]
        p_pre = p[:-5]
        rtn = np.log(p_now / p_pre)
        rtn_mean = np.nanmean(rtn)
        rtn_std = np.nanstd(rtn)
        Extre_num = rtn_mean - 1.96 * rtn_std
        Extre_fre = rtn - Extre_num
        Extre_fre = Extre_fre < 0
        df["exRtn_minFre"] = np.nansum(Extre_fre)
        return df
    df = df.groupby("time_id").apply(cal_exRtn_minFre).reset_index(drop=True)
    return df
