# Project for automatic high-frequency trading of BTC

## Approach:

Price movement is treated as a binary classification problem. Data samples are a time-series of bitcoin price data in 1m resolution. Labels are future prices (15m or 30m interval) being higher/lower. Prediction model is a time-series transformer with a preceding temporal convolution block for feature generation. Additionally, uncertainty estimates for the predictions are calculated via the [Laplace approximation](https://arxiv.org/abs/2106.14806).
Predictive accuracy is up to 60-70%.


Trading strategy (when to enter trade, how to set take-profit/stop-loss, etc...) was realized either through fixed rules or through a genetic algorithm. Tradings signals are the predictions from classification model.


Strategy was tested on simulated trading environment, including all necessary details for analysis of viability of the algorithm, like spread, slippage, fees, triggering of TP/SL, trailing SL.


### Test results:

In simplified setting (no fees, no slippage) the algorithm yields consistent profits over time, outperforming any longterm trading strategy. However, with fees and slippage the algorithm fails to produce a positive result. Main observation is, that, despite the majority of predictions being correct (60-70% classification accuracy), rare extreme swings in price cause large losses, that outweight the impact of the profitable trades. This is due to the large slippage occuring during these trades, where the SL is triggered and the subsequent market order is executed way below SL level.
