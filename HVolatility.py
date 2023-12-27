import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import yfinance as yfin
from matplotlib import pyplot

startDate = '2020-09-01'
endDate = '2021-07-01'
stockSymbol = 'BTC-USD'
fileName = 'downloadedData.pk1'

time_period = 7 # day window ( REQUIREMENT 2)
alpha = 2/(time_period + 1)

yfin.pdr_override()
data = yfin.download(tickers=stockSymbol, start=startDate, end=endDate)

closingPrice = data['Adj Close']

# Make the Returns time series

returns = []
for i in range(1, len(closingPrice)):
    daily_return = (closingPrice[i] -closingPrice[i-1])/closingPrice[i-1] * 100
    returns.append(daily_return)

# calculate moving avg (REQUIREMENT 1)
exp_moving_avg = []
exp_moving_avg_value = 0

for price in returns:
    if exp_moving_avg_value == 0:
        exp_moving_avg_value = price
    else:
        exp_moving_avg_value = alpha*price+(1-alpha)*exp_moving_avg_value
    exp_moving_avg.append(exp_moving_avg_value)
data = data.assign(Daily_returns=pd.Series(returns, index= data.index[1::]))
data = data.assign(Exponential_moving_average=pd.Series(exp_moving_avg, index= data.index[1::])) # this is the time series

# # verify ewm
#
# exponential_moving_average_pandas=data['Adj Close'].ewm(span=time_period, adjust=False).mean()
#
# fig2 = plt.figure(figsize=(10,8))
# ax2 = fig2.add_subplot(111,ylabel='Moving Averages ($)')
# data['Exponential_moving_average'].plot(ax=ax2, color='r', lw=10, label='Manually Computed', legend=True)
# exponential_moving_average_pandas.plot(ax=ax2, color='k', lw=3, label='Pandas function', legend=True)
# plt.show()

def getTEvents(gRaw,h): # fcn as per the 2018 text
    tEvents,sPos,sNeg=[],0,0
    diff=gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

anomalies = getTEvents(data['Exponential_moving_average'], h=2) # REQUIREMENT 3
# reasonable threshold

# plot the series and anomalies
fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(111,ylabel='Daily Returns (%)')
ax = data['Exponential_moving_average'].plot(ax=ax1, color='r', lw=3, legend=True)
ax.scatter(anomalies, data['Exponential_moving_average'].loc[anomalies])
plt.show()

### SETUP FOR TRIPLE BARRIER
####################

def compute_vol(df, time_period): # keep time period consistent with previous
    df.fillna(method='ffill', inplace=True)
    r = df.pct_change()
    return r.ewm(span=time_period).std() # use pandas ewm

def triple_barrier_labels(df, t, time_period, upper=None, lower=None, dev=2.5):
    # df is column of data (volatility series), let it be one column only
    # t is future time frame window (assume t >= 1)
    # upper/lower are return limits
    # dev is standard deviation
    df.fillna(method='ffill', inplace=True)
    labels = pd.DataFrame(index=df.index, columns=['Label']) # initialize labels dataframe

    r = range(len(df)-1-t) # iterate and label all events
    for idx in r:
        s = df.iloc[idx:idx+t] # get window

        if not all(np.isfinite(s.cumsum().values)): # check for discontoinuity
            labels['Label'].iloc[idx] = np.nan
            continue

        # calculations for limit if not preset
        vol = compute_vol(df[:idx+t], time_period)
        if upper == None:
            u = vol.iloc[idx]*1.5*dev # more lenient upper, (1.5x range) asymmetrical (REQUIREMENT 5)
        else:
            u = upper
        if lower == None:
            l = -vol.iloc[idx]*dev
        else:
            l = lower

        if any(s.values >= u):
            labels['Label'].iloc[idx] = 1
        elif any(s.values <= l):
            labels['Label'].iloc[idx] = -1
        else:
            labels['Label'].iloc[idx] = np.sign(df.tolist()[idx]) # return sign of cur rather than zero (REQUIREMENT 6)
    df = pd.concat([df, labels], axis=1)
    return df

# you could just label the anomaly points too for better storage/runtime as shown below
tempanomalies = data['Exponential_moving_average']
labelled = triple_barrier_labels(tempanomalies, 30, time_period) # REQUIREMENT 4
# 30 day change for triple barrier detection

### VISUALIZATION - REQUIREMENT 7

anomaly_markers = labelled.loc[anomalies]
# print(anomaly_markers) # some of these will be NAN because the time window will clip end of data (anything after 2021-05-30)
# graph this jawn

# Good plot
x = anomaly_markers.index
y = anomaly_markers['Label']
pyplot.scatter(x,y)
pyplot.show()





