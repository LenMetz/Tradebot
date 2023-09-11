import numpy as np
import torch

class Trade:
    def __init__(self, orderId, openTime, side, openPrice, openAmount, takeProfit, stopLoss, trailingStopLoss, leverage, trailing):
        self.orderId = orderId
        self.openTime = openTime
        if side>0:
            self.side = "Long"
        else:
            self.side = "Short"
        self.openPrice = openPrice
        self.openAmount = openAmount
        self.takeProfit = takeProfit
        self.stopLoss = stopLoss # price at which stop loss is triggered
        self.trailingStopLoss = trailingStopLoss # decimal change to adjust stop loss
        self.leverage = leverage
        self.trailing = trailing
        
    # close trade
    def close(self):
        self.status = 0
        self.side = None

    # adjust stop loss if trailing stop loss passes it
    def updateStopLoss(self, price):
        #print(price, self.openPrice, self.stopLoss, price+price*self.trailingStopLoss/self.leverage, self.side)
        if self.side=="Long":
            self.stopLoss = max(self.stopLoss, price+price*self.trailingStopLoss/self.leverage)
        if self.side=="Short":
            self.stopLoss = min(self.stopLoss, price-price*self.trailingStopLoss/self.leverage)
        #print(self.stopAdjusted)
        
    def step(self, price, priceLow, priceHigh):
        #print(self.side, self.openPrice, self.stopLoss, self.takeProfit, price, priceLow, priceHigh)
        #print(price, priceLow, priceHigh, self.stopLoss, )
        if self.side=="Long":
            if priceLow>=self.takeProfit:
                #print("long prof")
                return "Take profit", round(self.openAmount*(priceLow-self.openPrice), 3), priceLow
            if priceLow<=self.stopLoss:
                #print("long loss")
                return "Stop loss", round(self.openAmount*(priceLow-self.openPrice), 3), priceLow
        if self.side=="Short":
            if priceHigh<=self.takeProfit:
                #print("short prof")
                return "Take profit", round(-self.openAmount*(priceHigh-self.openPrice), 3), priceHigh
            if priceHigh>=self.stopLoss:
                #print("short loss")
                return "Stop loss", round(-self.openAmount*(priceHigh-self.openPrice), 3), priceHigh
        if self.trailing:
            self.updateStopLoss(price)
        return "hold", 0, 0







