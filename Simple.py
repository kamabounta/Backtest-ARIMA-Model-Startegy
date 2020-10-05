import backtrader as bt 
import math
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class my_strat(bt.Strategy):
    
    params = (('max_position', 10), )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.signal = ARIMA_ind(self.dataclose)
        
    def log(self, txt, dt=None):
        ''' 
        Logging function for this strategy
        '''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
		
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
		# Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

		# Check if an order has been completed
		# Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            
            self.log('Order Canceled/Margin/Rejected')

		# Write down: no pending order
        self.order = None
        
    def next(self):

        if self.signal > 0:
            if self.position.size < self.params.max_position:
                self.buy()

        elif self.signal < 0:
            if self.position.size > 0:
                self.close()  
        
class ARIMA_ind(bt.Indicator):
    
    lines = ('ARIMA_Model_Returns_Forecast', )
    params = (('period', 20), )

    plotinfo = dict(
        plot=True,
        plotname='ARIMA_Model_Returns_Forecast',
        subplot=True,
        plotlinelabels=True)
    
    def __init__(self):
        self.addminperiod(self.params.period)
    
    def next(self):
        x = self.data.get(size = self.p.period)
        # percent returns
        X = [(a / x[x.index(a) - 1]) - 1 for a in x]
        size = int(len(X) * 0.8)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order = (0, 1, 1))
            model_fit = model.fit(disp = -1)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print ('predicted = %f, expected = %f'%(yhat, obs))
        
        self.lines.ARIMA_Model_Returns_Forecast[0] = predictions[-1]
        
        """
        Sharpe Ratio: OrderedDict([('sharperatio', 0.5928765212729156)])
        Final Portfolio Value: 1374.18
        """
        
class auto_ARIMA_ind(bt.Indicator):
    
    lines = ('ARIMA_Model_Returns_Forecast', )
    params = (('period', 30), )

    plotinfo = dict(
        plot=True,
        plotname='ARIMA_Model_Returns_Forecast',
        subplot=True,
        plotlinelabels=True)
    
    def __init__(self):
        self.addminperiod(self.params.period)
    
    def next(self):
        x = self.data.get(size = self.p.period)
        # percent returns
        X = [(a / x[x.index(a) - 1]) - 1 for a in x]
        size = int(len(X) * 0.8)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            #Automatically selects the best parameters for ARIMA model
            model = model = pm.auto_arima(history, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    
            model_fit = model.fit(disp = -1)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print ('predicted = %f, expected = %f'%(yhat, obs))
        
        self.lines.ARIMA_Model_Returns_Forecast[0] = predictions[-1]
        
        """
        Sharpe Ratio: OrderedDict([('sharperatio', 0.7088414735506292)])
        Final Portfolio Value: 1567.80
        """