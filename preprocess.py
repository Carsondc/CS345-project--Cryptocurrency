import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
# list of all files the user can choose from. This will be chosen from the user front end
potential_files = [
    "1INCH.csv",
    "AAVE.csv",
    "ADA.csv",
    "ALGO.csv",
    "AMP.csv",
    "APE.csv",
    "AR.csv",
    "ATOM.csv",
    "AVAX.csv",
    "AXS.csv",
    "BAT.csv",
    "BCH.csv",
    "BNB.csv",
    "BSV.csv",
    "BTC.csv",
    "BTT.csv",
    "CAKE.csv",
    "CFX.csv",
    "CHZ.csv",
    "COMP.csv",
    "CRO.csv",
    "CRV.csv",
    "CVX.csv",
    "DAI.csv",
    "DASH.csv",
    "DCR.csv",
    "DGB.csv",
    "DOGE.csv",
    "DOT.csv",
    "DYDX.csv",
    "EGLD.csv",
    "ENS.csv",
    "ETC.csv",
    "ETH.csv",
    "FET.csv",
    "FIL.csv",
    "FLOW.csv",
    "FTT.csv",
    "GALA.csv",
    "GLM.csv",
    "GRT.csv",
    "GT.csv",
    "HBAR.csv",
    "HNT.csv",
    "ICP.csv",
    "IMX.csv",
    "INJ.csv",
    "IOTA.csv",
    "JST.csv",
    "KCS.csv",
    "KSM.csv",
    "LDO.csv",
    "LEO.csv",
    "LINK.csv",
    "LPT.csv",
    "LTC.csv",
    "MANA.csv",
    "MINA.csv",
    "MX.csv",
    "NEAR.csv",
    "NEO.csv",
    "NEXO.csv",
    "NFT.csv",
    "OKB.csv",
    "PAXG.csv",
    "QNT.csv",
    "QTUM.csv",
    "RAY.csv",
    "RSR.csv",
    "RUNE.csv",
    "SAND.csv",
    "SHIB.csv",
    "SNX.csv",
    "SOL.csv",
    "STX.csv",
    "SUN.csv",
    "SUPER.csv",
    "SYRUP.csv",
    "THETA.csv",
    "TWT.csv",
    "UNI.csv",
    "USDC.csv",
    "USDD.csv",
    "USDT.csv",
    "VET.csv",
    "WEMIX.csv",
    "XAUt.csv",
    "XCN.csv",
    "XEC.csv",
    "XLM.csv",
    "XMR.csv",
    "XNO.csv",
    "XRP.csv",
    "XTZ.csv",
    "ZEC.csv",
    "ZEN.csv",
    "ZRX.csv",
]
def sma(array, window):
    close = array[:, 3]                     
    kernel = np.ones(window) / window
    sma_vals = np.convolve(close, kernel, mode='valid')
    return sma_vals
    #simple moving average, finance thing
def golden_cross(array):
    sma50  = sma(array, 50)
    sma200 = sma(array, 200)
    sma50 = sma50[-len(sma200):]
    above = sma50 > sma200
    crossed = (above[1:] & ~above[:-1])  
    return crossed.astype(int)
#golden cross means price will increase chi ching!
    
def death_cross(array):
    sma50  = sma(array, 50)
    sma200 = sma(array, 200)
    sma50 = sma50[-len(sma200):]
    above = sma50 > sma200
    crossed = (~above[1:] & above[:-1])  
    return crossed.astype(int)
def generate_y(array, horizon):
    close = array[:,4]
    return close[horizon:] # horizon is how many days in advance we are predicting
#golden and death cross are finance stuff that can predict if sales will increase or decrease.
class preprocess:
    def __init__(self,files,horizon):
            self.horizon = horizon
            self.files = files
    def generate_data(self): #constructor for generate data. Both values will be populated from user input.
        files = self.files
        horizon = self.horizon
        pre_processed_datasets = [] #list containing all x,y tuples
        print("processing files")
        for file_path in files:
            print(f"Loading {file_path}...")
            df= kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "svaningelgem/crypto-currencies-daily-prices",
                file_path, #built in kaggle function creates a pandas data frame fromm kaggle data
            )
            array = df.to_numpy()
            array = array[1:, 2:] #skip first row, remove sticker and time stamp
            horizon
            sma_vals = sma(array, horizon).reshape(-1, 1)
            array = array[horizon-1:, :]     
            array = np.hstack([array, sma_vals]) #adding sma vlaues column
            golden_cross_values = golden_cross(array).reshape(-1,1) #golden cross was a 2d array, needs to be 1d
            death_cross_values = death_cross(array).reshape(-1,1) #death cross was a 2d array, needs to be 1d
            array = array[200:] #shift values from cross
            array = np.hstack([array,golden_cross_values]) #adding golden cross values column
            array = np.hstack([array,death_cross_values]) #adding death cross values column
            prefix = np.arange(len(array)).reshape(-1, 1) #one id array of values that can be hstacked
            array = np.hstack([prefix, array]) #the prefix is the number of days since the start of the data set.
            y = generate_y(array,horizon)
            X = array[:-horizon] #shortening X to match up with y
            X_y_tuple = (X,y) #This is a tuple of the 2d X array and thr 2d y array
            pre_processed_datasets.append(X_y_tuple) #This list containes tuples of each data set
        print("The columns are: (number of days since start,open,high,low,close,sma(10 days),golden cross, death cross)")
        return pre_processed_datasets # The result is a list of pre-processed X,y values for every dataset created by the user.
if __name__ == "__main__":
    processor = preprocess(potential_files[:3], 7)
    list_of_lists = processor.generate_data()
    X2 = list_of_lists[1][0]
    Y2 = list_of_lists[1][1]
    print(X2[205])
    print(Y2[198])



