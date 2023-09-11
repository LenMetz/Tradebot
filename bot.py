import torch
import numpy as np
import pickle
from trade import Trade
from server import Server
from model import Temporal_Conv_Transformer, Temporal_Conv_Transformer_Vol
import matplotlib.pyplot as plt
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OfflineBot:
    def __init__(self, inputs):
        self.inputs = inputs
        file = open(f'configs/tradeConfig_{inputs.configId}', 'rb')
        self.config = pickle.load(file)
        file.close()
        self.model = Temporal_Conv_Transformer_Vol(seq_length = self.config["sequenceLength"], feature_size=self.config["featureSize"],
                                          dropout=self.config["dropout"], num_layers=self.config["numLayers"])
        #self.model2 = Temporal_Conv_Transformer_Vol(seq_length = self.config["sequenceLength"], feature_size=self.config["featureSize"],
                                          #dropout=self.config["dropout"], num_layers=self.config["numLayers"])
        #self.model3 = Temporal_Conv_Transformer_Vol(seq_length = self.config["sequenceLength"], feature_size=self.config["featureSize"],
                                          #dropout=self.config["dropout"], num_layers=self.config["numLayers"])
        self.model.load_state_dict(torch.load("model/" + inputs.modelId+".pt", map_location=device))
        #self.model2.load_state_dict(torch.load("model/" + inputs.modelId2+".pt", map_location=device))
        #self.model3.load_state_dict(torch.load("model/" + inputs.modelId3+".pt", map_location=device))
        self.model.to(device)
        #self.model2.to(device)
        #self.model3.to(device)
        self.server = Server()
        self.logfile = f"logs/tradelogs{inputs.name}_mIds{inputs.modelId}_cId{inputs.configId}_sPr{inputs.slip}.txt"
        s1, s2, s3, s4, s5, s6, s7, s8, s9  = "ID", "Open Time", "Close Time", "Side", "Amount",\
        "Open Price", "Close Price", "Close Type", "Capital Change"
        with open(self.logfile, "w") as f:
            f.write("\n## START ### \n\n" + f"{s1:<15} | {s2:<15} | {s3:<15} | {s4:<15} | {s5:<15} | {s6:<15} | {s7:<15} | {s8:<15} | {s9:<15}")
        self.prices = []
        self.volumes = []
        self.capital = 1000
        self.timeStep = 0
        self.timeStepSecond = 0
        self.currentMin = None
        self.trade = None
        self.tradeNum = 0
        self.tradeProb = 1
        self.slipProb = inputs.slip
        self.cooldown = 0
        self.capitalOverTime = []
        print("Model ID: ", inputs.modelId)
        self.tgtStep = self.config["tgtStep"]
        self.tradeLens = 0
        self.lastTradeNum = 0
        self.profits = 0
        self.losses = 0
        self.t = 0
        self.f = 0


    def saveTrade(self, tradeNum, openTime, closeTime, side, openAmount, openPrice, exitPrice, closeType, change):
        with open(self.logfile, "a") as f:
            f.write(f"\n{tradeNum:>15} | {openTime:>15} | {closeTime:>15} | {side:>15} | {openAmount:>15}  | {openPrice:>15} | {exitPrice:>15} | {closeType:>15} | {change:>15}")
            
    def open(self, priceClose):
        draw = np.random.binomial(1,self.tradeProb,1)
        
        if draw and self.cooldown<=0:
            src = (torch.tensor(self.prices[-self.config["sequenceLength"]-1:]).unsqueeze(dim=0)-self.prices[-1]).float()
            vol = (torch.tensor(self.volumes[-self.config["sequenceLength"]-1:]).unsqueeze(dim=0)).float()
            vol = vol/torch.max(vol)
            #print(self.server.getTgt(self.timeStepSecond, self.tgtStep), self.prices[-1])
            tgt = torch.tensor(self.server.getTgt(self.timeStepSecond, self.tgtStep)-self.prices[-1])
            eps = torch.zeros(src.size())
            with torch.no_grad():
                pred, mu, std = self.model(src, vol, eps)
            #mu = -mu
            #mu = torch.bernoulli(torch.tensor(0.5))*(-2)+1
            
                
            if torch.abs(mu)>=self.config["muClip"]*self.config["maxMu"] and torch.abs(std)<=self.config["stdClip"]*self.config["maxStd"]:
                if torch.sign(mu)==torch.sign(tgt):
                    self.t +=1
                else:
                    self.f +=1
                self.tradeNum += 1
                side = int(torch.sign(mu))
                openAmount = self.config["invest"]*self.config["leverage"]*self.capital/priceClose
                openAmount = round(openAmount - openAmount*(self.config["fees"]+self.config["spread"]),3)
                takeProfit = round(priceClose + side*self.config["takeProfit"]*priceClose/self.config["leverage"], 2)
                stopLoss = round(priceClose + side*self.config["stopLoss"]*priceClose/self.config["leverage"], 2)
                #print(side, takeProfit, stopLoss, priceClose)
                
                self.trade = Trade(f"{self.tradeNum:09d}", self.timeStep, side, priceClose,
                                   openAmount, takeProfit, stopLoss, self.config["trailingStopLoss"], self.config["leverage"], self.inputs.trailing)
                #print(self.trade.openPrice, self.trade.stopLoss, self.trade.takeProfit)
    def close(self, priceClose, priceLow, priceHigh):
        result, change, exitPrice = self.trade.step(priceClose, priceLow, priceHigh)
        
        if result!="hold":
            #print(result, change, self.capital)
            slip = np.random.binomial(1,self.slipProb,1)[0]
            
            #if (exitPrice-self.trade.openPrice)*(-int(self.trade.side=="Short")+int(self.trade.side=="Long"))<0:
            if result=="Stop loss":
                #print("\n", change)
                change = slip*change + (1-slip)*self.trade.openAmount*(self.trade.stopLoss-self.trade.openPrice)*\
                (-int(self.trade.side=="Short")+int(self.trade.side=="Long"))
                #print(change, slip, self.trade.side, self.trade.openPrice, self.trade.stopLoss, exitPrice)
                self.capital += change - self.config["invest"]*self.capital*(self.config["fees"]+self.config["spread"])*self.config["leverage"]
                if change<0:
                    self.cooldown = self.config["cooldown_reset_l"]
                    self.losses += 1
                    self.tradeProb = self.config["penalty"]*self.tradeProb
                else:
                    self.profits += 1
            else:
                change = round(self.trade.openAmount*(self.trade.takeProfit-self.trade.openPrice)*\
                               (-int(self.trade.side=="Short")+int(self.trade.side=="Long")), 3)
                self.cooldown = self.config["cooldown_reset_p"]
                self.capital += change - self.config["invest"]*self.capital*(self.config["fees"]+self.config["spread"])*self.config["leverage"]
                self.profits += 1
                self.tradeProb = (3/self.config["penalty"])*self.tradeProb
                
            self.saveTrade(self.trade.orderId, self.trade.openTime, self.timeStep, self.trade.side, self.trade.openAmount,\
                           self.trade.openPrice, exitPrice, result, change)
            self.tradeLens += self.timeStep-self.trade.openTime
            self.trade = None
            

                
            
            #    if self.trade is not None:
            #       print(self.trade.openPrice, self.trade.stopLoss, self.trade.takeProfit, priceClose)

        self.capitalOverTime.append(self.capital)
        

    def run(self):
        maxLen = self.server.getLen()
        while self.timeStepSecond<maxLen-self.tgtStep*60:
            priceClose, priceLow, priceHigh, volume, t = self.server.getPriceData(self.timeStepSecond)
            
            if self.trade is not None:
                self.close(priceClose, priceLow, priceHigh)
                
            if t[14:16]!=self.currentMin:
                self.prices.append(priceClose)
                self.volumes.append(volume)
                self.currentMin = t[14:16]
                self.cooldown = self.cooldown - 1
                self.tradeProb += (1-self.tradeProb)/self.config["penaltyDecay"]
                self.tradeProb = min(1,self.tradeProb)
                
                if self.trade is None and self.timeStep>self.config["sequenceLength"]+self.tgtStep:
                    self.open(priceClose)
                    
                    if self.timeStep%(60*24*7)==0:
                        print("\n\n### Timestep: ", self.timeStep, " ###")
                        print("capital: ", self.capital)
                        print("p | l | p/p+l", self.profits, self.losses, self.profits/(self.profits+self.losses))
                        print("t | n | t/t+n", self.t,self.f, self.t/(self.t+self.f))
                        print(self.tradeLens/(self.tradeNum-self.lastTradeNum))
                        self.tradeLens, self.lastTradeNum = 0, self.tradeNum
                        #self.t=0
                        #self.f=0
                self.timeStep+=1
            
            #self.performTrade(priceClose, priceClose, priceClose)

            
            self.timeStepSecond +=1
            if self.capital<1:
                break
        plt.yscale("log")
        plt.plot(self.capitalOverTime)
        plt.savefig(f"figures/{inputs.name}_mId{inputs.modelId}_cId{inputs.configId}")

        
def parse_args():
    parser=argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument('-c', '--configId', default="00001")
    parser.add_argument('-m', '--modelId', default="001")
    parser.add_argument('-m2', '--modelId2', default="002")
    parser.add_argument('-m3', '--modelId3', default="003")
    parser.add_argument('-n', '--name', default="test")
    parser.add_argument('-t', '--trailing', action="store_true")
    parser.add_argument('-s', '--slip', default=1.0, type=float)
    args=parser.parse_args()
    return args


if __name__ == '__main__':
    inputs = parse_args()
    bot = OfflineBot(inputs)
    bot.run()
