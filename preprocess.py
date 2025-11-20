import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
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

# print(len(potential_files))

def time_stamp_to_days(array,time_stamp_index):
    for i in range(array.shape[0]):
        array[i][time_stamp_index] = i
def sma(array, window):
    close = array[:, 3]                     
    kernel = np.ones(window) / window
    sma_vals = np.convolve(close, kernel, mode='valid')
    return sma_vals
def golden_cross(array):
    sma50  = sma(array, 50)
    sma200 = sma(array, 200)
    sma50 = sma50[-len(sma200):]
    above = sma50 > sma200
    crossed = (above[1:] & ~above[:-1])  
    return crossed.astype(int)
    
def death_cross(array):
    sma50  = sma(array, 50)
    sma200 = sma(array, 200)
    sma50 = sma50[-len(sma200):]
    above = sma50 > sma200
    crossed = (~above[1:] & above[:-1])  
    return crossed.astype(int)
def generate_y(array, horizon):
    close = array[:,4]
    close_shape = close.shape
    close_horizon = close[horizon:]
    close_horizon_shape = close_horizon.shape
    return close[horizon:]
    # return close
class preprocess:
    def __init__(self,files,horizon):
            self.horizon = horizon
            self.files = files   
    def generate_data(self):
        files = self.files
        horizon = self.horizon
##### OLD START ###############################################################################################
        # pre_processed_datasets = []
##### OLD END ###############################################################################################
##### NEW START ###############################################################################################
        pre_processed_datasets = {}
##### NEW END ###############################################################################################
        print("processing files")
##### OLD START ###############################################################################################
        # for file_path in files:
##### OLD END ###############################################################################################
        for index, file_path in enumerate(files):
##### NEW START ###############################################################################################
            print(f"Loading {file_path}...")
            
            try:
                df= kagglehub.dataset_load(
                    KaggleDatasetAdapter.PANDAS,
                    "svaningelgem/crypto-currencies-daily-prices",
                    file_path,
                )
                print('its working')
            except Exception as e:
                print(f'failed to load {file_path} : {e}')
            
            array = df.to_numpy()
            array = array[1:, 2:]
            window = 10
            sma_vals = sma(array, window).reshape(-1, 1)
            array = array[window-1:, :]     
            array = np.hstack([array, sma_vals])
            prefix = np.arange(len(array)).reshape(-1, 1)
            array = np.hstack([prefix, array])
            golden_cross_values = golden_cross(array).reshape(-1,1)
            death_cross_values = death_cross(array).reshape(-1,1)
            array = array[200:] #shift values from cross
            array = np.hstack([array,golden_cross_values])
            array = np.hstack([array,death_cross_values])
            prefix = np.arange(len(array)).reshape(-1, 1)
            array = np.hstack([prefix, array])
            
            X = array[:, :-horizon]
            X_shape = X.shape
            y = generate_y(array,horizon)
            y_shape = y.shape
##### OLD START ###############################################################################################
            # X_y_tuple = (X,y)
            # pre_processed_datasets.append(X_y_tuple)
##### OLD END ###############################################################################################
##### NEW START ###############################################################################################
            ## using nested dictionaries so that we have O(1) average case time for look ups
            xy_data_for_file_path = {'X' : X, 'y' : y}
            pre_processed_datasets[file_path] = xy_data_for_file_path
            
##### NEW START ###############################################################################################
        print("The features are:\nnumber of days since start | open | high | low | close | sma(10 days) | golden cross | death cross")
        return pre_processed_datasets

## code below will only execute when ran from command line rather than imported
if __name__ == "__main__":
    processor = preprocess(potential_files,7)
    data = processor.generate_data()
    print(data)
    



