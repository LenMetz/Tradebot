import numpy as np
import torch


class Server:
    def __init__(self):
        '''
        self.pricesClose = np.load(f"data/2023test1m.npy")
        self.pricesLow = np.load(f"data/2023test1m_low.npy") 
        self.pricesHigh = np.load(f"data/2023test1m_high.npy")'''
        
        self.pricesClose = np.load(f"data/BINANCE_PERP_2023_1SEC_close.npy")
        self.pricesLow = np.load(f"data/BINANCE_PERP_2023_1SEC_low.npy") 
        self.pricesHigh = np.load(f"data/BINANCE_PERP_2023_1SEC_high.npy")
        self.Volume = np.load(f"data/BINANCE_PERP_2023_1SEC_volume.npy")
        self.tEnd = np.load(f"data/BINANCE_PERP_2023_1SEC_t_end.npy")

    # retuns price data for given timestep (Close, Low, High)
    def getPriceData(self, timeStep):
        return self.pricesClose[timeStep], self.pricesLow[timeStep], self.pricesHigh[timeStep], self.Volume[timeStep], self.tEnd[timeStep]

    def getTgt(self, timeStepSeconds, tgtStep):
        return self.pricesClose[timeStepSeconds+tgtStep*60]

    def getLen(self):
        return len(self.pricesClose)
        