import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.switch_backend('Qt5Agg')

data = pd.read_csv('msft.csv')
data.set_index('Date', inplace = True)
ts = data['Close'].pct_change()
ts = ts[1:]

def one_period_simple_return(data):
    
    data['simple_return'] = data['Close'] / data['Close'].shift(1, axis = 0) - 1
    # data['simple_return'] = data['Close'].pct_change()
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:purple'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Returns', color=color)
    ax1.plot(data.simple_return[1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Close Price', color=color)  # we already handled the x-label with ax1
    ax2.plot(data.Close[1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('One Period Simple Returns')
    plt.show()
    
def multi_period_simple_return(data, period):
    
    data['multi_period_return'] = data['Close'] / data['Close'].shift(period, axis = 0) - 1
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:purple'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Returns', color=color)
    ax1.plot(data.index[1:], data.multi_period_return[1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Close Price', color=color)  # we already handled the x-label with ax1
    ax2.plot(data.index[1:], data.Close[1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Multi Period Simple Returns : %.0f days' % period)
    plt.show()    
    
def continuously_compounded_return(data):
    
    data['cc_return'] = np.log(1 + (data['Close'] / data['Close'].shift(1, axis = 0)))
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:purple'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Returns', color=color)
    ax1.plot(data.index[1:], data.cc_return[1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Close Price', color=color)  # we already handled the x-label with ax1
    ax2.plot(data.index[1:], data.Close[1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Continuously Compounded Returns')
    plt.show()
    
def distributional_properties(data):
    
    data['simple_return'] = data['Close'].pct_change()
    
    mu, sigma = data['simple_return'].mean(), data['simple_return'].std()
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(data['simple_return'], 50, density = True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    plt.title('Distributional Properties of Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()
    
    print (stats.describe(data.simple_return[1:]))

def skewness(data):
    data['simple_return'] = data['Close'].pct_change()
    mean = data.simple_return[1:].mean()
    std = data.simple_return[1:].std()
    n = len(data.simple_return[1:])
    sigma = 0
    for i in range(1, n):
        sigma += ((data.simple_return[i] - mean) / (std)) ** 3
    skew = (sigma * n)/((n - 1) *  (n - 2))
    print ('Skew :', skew)
    
def autocorrelation_returns(data, return_type):
    
    data['simple_return'] = data['Close'].pct_change()
    data['cc_return'] = np.log(1 + (data['Close'] / data['Close'].shift(1, axis = 0))) #log return
    
    if return_type == 'simple':
        # plt.acorr(data.simple_return[1:], maxlags = 20)
        plot_acf(data.simple_return[1:])
        plt.title('Autocorrelation of Simple Returns')
        plt.xlabel('Lags')
        plt.ylabel('Correlation')
        print (sm.stats.acorr_ljungbox(data.simple_return[1:], lags=[30], return_df=True))
        if sm.stats.acorr_ljungbox(data.simple_return[1:], lags=[30], return_df=False)[1] < 0.0001:
            print ('Simple Returns are serially correlated')
        else:
            print ('Simple Returns are not serially correlated')
        
    elif return_type == 'log':
        # plt.acorr(data.cc_return[1:], maxlags = 20)
        plot_acf(data.cc_return[1:])
        plt.title('Autocorrelation of Log Returns')
        plt.xlabel('Lags')
        plt.ylabel('Correlation')
        print (sm.stats.acorr_ljungbox(data.cc_return[1:], lags=[30], return_df=True))
        if sm.stats.acorr_ljungbox(data.simple_return[1:], lags=[30], return_df=False)[1] < 0.0001:
            print ('Log Returns are serially correlated')
        else:
            print ('Log Returns are not serially correlated')
            
    """
               lb_stat     lb_pvalue
    30  312.105175  1.079277e-48
    Simple Returns are serially correlated
           lb_stat     lb_pvalue
    30  309.635054  3.323054e-48
    Log Returns are serially correlated
    """

def correlation(data):
    
    data['simple_return'] = data['Close'].pct_change()
    data['lagged'] = data['simple_return'].shift(1, axis = 0)
    plt.scatter(data.lagged[2:], data.simple_return[2:])
    plt.title('Correlation between return and 1-day-lag return') 
    plt.xlabel('X[t-1]')
    plt.ylabel('X[t+1]')
    
    dataframe = pd.concat([data.lagged[2:], data.simple_return[2:]], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()
    print (result)
    plt.show()
    
    """
              t-1       t+1
    t-1  1.000000 -0.311883
    t+1 -0.311883  1.000000
    """

def stationarity_check(ts): 
    
    # 1. Check Stationarity (moving mean/variance should not move with time)
    rolmean = ts.rolling(20).mean()
    rolstd = ts.rolling(20).std()
    original = plt.plot(ts, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'upper left')
    plt.title('Rolling Mean and Standard Deviation of Returns')
    plt.show(block = False)
    
    '''
    The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.

    There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. 
    It uses an autoregressive model and optimizes an information criterion across multiple different lag values.

    The null hypothesis of the test is that the time series can be represented by a unit root, that it is not 
    stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is 
    that the time series is stationary.

    We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) 
    suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we 
    fail to reject the null hypothesis (non-stationary).

    p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
    p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    
    '''

    #2. Dicky Fuller Test to test Stationarity
    print ('Results of Dicket Fuller Test :')
    dftest = adfuller(ts, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
    #We can see that our statistic value of -11.20287 is less than the value of -3.449 at 1%.
    #This suggests that we can reject the null hypothesis with a significance level of 
    #less than 1%.
    #This means that the process does not have a unit root, and in turn that the time series is 
    #stationary or does not have time-dependent structure.
    
    """
    Results of Dicket Fuller Test :
    Test Statistic                -1.120287e+01
    p-value                        2.221409e-20
    #Lags Used                     8.000000e+00
    Number of Observations Used    9.900000e+02
    Critical Value (1%)           -3.436973e+00
    Critical Value (5%)           -2.864464e+00
    Critical Value (10%)          -2.568327e+00
    dtype: float64
    """
    
def eliminating_trend(ts):
    
    '''
    The underlying principle is to model or estimate the trend and seasonality in the 
    series and remove those from the series to get a stationary series. Then statistical
    forecasting techniques can be implemented on this series. The final step would be to 
    convert the forecasted values into the original scale by applying trend and seasonality 
    constraints back
    
    In this simpler case, it is easy to see a forward trend in the data. But its not very 
    intuitive in presence of noise. So we can use some techniques to estimate or model this 
    trend and then remove it from the series. There can be many ways of doing it and some of 
    most commonly used are:

    Aggregation – taking average for a time period like monthly/weekly averages
    Smoothing – taking rolling averages
    Polynomial Fitting – fit a regression model

    '''
    log_ts = np.log(ts + 1)
    # moving_avg = log_ts.rolling(20).mean()
    weighted_avg = np.average(log_ts)
    # plt.plot(log_ts)
    # plt.plot(moving_avg, color='red')
    
    weighted_avg_diff = log_ts - weighted_avg
    weighted_avg_diff.dropna(inplace = True)
    stationarity_check(weighted_avg_diff)
    
    """
    Results of Dicket Fuller Test :
    Test Statistic                -1.108386e+01
    p-value                        4.244721e-20
    #Lags Used                     8.000000e+00
    Number of Observations Used    9.900000e+02
    Critical Value (1%)           -3.436973e+00
    Critical Value (5%)           -2.864464e+00
    Critical Value (10%)          -2.568327e+00
    dtype: float64
    """
    
    #This looks like a good series. The rolling values appear to be varying slightly 
    #but there is no specific trend. Also, the test statistic is smaller than the 1% critical 
    #values so we can say with 99% confidence that this is a stationary series.
    #However, a drawback in this particular approach is that the time-period has to be strictly defined.
    # We take a ‘weighted moving average’ where more recent 
    #values are given a higher weight. In this case there will be no missing values as all values from
    #starting are given weights. So it’ll work even with no previous values.
    
def differencing(ts):
    
    '''
    The simple trend reduction techniques discussed before don’t work in all cases, particularly the ones 
    with high seasonality. Lets discuss two ways of removing trend and seasonality:

    Differencing – taking the differece with a particular time lag
    Decomposition – modeling both trend and seasonality and removing them from the model.
    '''
    
    #One of the most common methods of dealing with both trend and seasonality is differencing. 
    #In this technique, we take the difference of the observation at a particular instant with 
    #that at the previous instant. This mostly works well in improving stationarity. 
    
    log_ts = np.log(ts + 1)
    ts_log_diff = log_ts - log_ts.shift(1)
    plt.plot(ts_log_diff)    
    
    ts_log_diff.dropna(inplace = True)
    stationarity_check(ts_log_diff)
    
    """
    Results of Dicket Fuller Test :
    Test Statistic                -1.410245e+01
    p-value                        2.604226e-26
    #Lags Used                     1.600000e+01
    Number of Observations Used    9.810000e+02
    Critical Value (1%)           -3.437033e+00
    Critical Value (5%)           -2.864491e+00
    Critical Value (10%)          -2.568341e+00
    dtype: float64
    """
   
def decomposing(ts):
    
    log_ts = np.log(ts + 1)
    decomposition = seasonal_decompose(log_ts, freq = 20)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(log_ts, label = 'Original')
    plt.legend(loc = 'best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    
    #Here we can see that the trend, seasonality are separated out from data and we can model the residuals. 
    #Lets check stationarity of residuals:
        
    
    log_data_decompose = residual
    log_data_decompose.dropna(inplace = True)
    stationarity_check(log_data_decompose)
    
    """
    Results of Dicket Fuller Test :
    Test Statistic                -1.205855e+01
    p-value                        2.500315e-22
    #Lags Used                     1.900000e+01
    Number of Observations Used    9.590000e+02
    Critical Value (1%)           -3.437187e+00
    Critical Value (5%)           -2.864559e+00
    Critical Value (10%)          -2.568377e+00
    dtype: float64
    """
    
    #The Dickey-Fuller test statistic is significantly lower than the 1% critical value. So this TS is very close to stationary.

    
def acf_pcf_plot(ts):
    
    '''
    ARIMA stands for Auto-Regressive Integrated Moving Averages. 
    The ARIMA forecasting for a stationary time series is nothing but a linear (like a linear regression) equation. 
    The predictors depend on the parameters (p,d,q) of the ARIMA model:

    Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).
    Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t)
    will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.
    Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first order difference. 
    So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.

    An importance concern here is how to determine the value of ‘p’ and ‘q’. We use two plots to determine these numbers.

    1. Autocorrelation Function (ACF)
    2. Partial Autocorrelation Function (PACF)
    '''
    
    log_ts = np.log(ts + 1)
    ts_log_diff = log_ts - log_ts.shift()
    ts_log_diff.dropna(inplace = True)
    
    lag_acf = acf(ts_log_diff, nlags = 20)
    lag_pacf = pacf(ts_log_diff, nlags = 20, method = 'ols')
    
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y = 0, linestyle = '--', color = 'gray')
    plt.axhline(y = -1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
    plt.axhline(y = 1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
    plt.title('Autocorrelation Function')
    
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y = 0, linestyle = '--', color = 'gray')
    plt.axhline(y = -1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
    plt.axhline(y = 1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
    """
    In this plot, the two dotted lines on either sides of 0 are the confidence interevals. These can be used to determine the ‘p’ and ‘q’ values as:

    p – The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case p=1.
    q – The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case q=1
    """
    
def AR_model(ts):
    
    """
    ACCURACY METRICS :=
        mean absolute percentage error : Around 2.2% MAPE implies the model is 
        about 97.8% accurate in predicting the next 15 observations. Around 2.2% 
        MAPE implies the model is about 97.8% accurate in predicting the next 15 observations.

            mape = np.mean(np.abs(predictions[i] - test[i])/np.abs(test[i]) for i in range(len(test))) 
            print ('Test MAPE: %.6f'%mape)
        correlation coefficient
            corr = np.corrcoef(predictions, test)[0,1]
            print ('Correlation Coefficient : %.6f'%corr)
    """
    log_ts = np.log(ts + 1)
    ts_log_diff = log_ts - log_ts.shift()
    ts_log_diff.dropna(inplace = True)
    
    model = ARIMA(log_ts, order = (1, 1, 0))
    results_AR = model.fit(disp = -1)
    RSS = sum((results_AR.fittedvalues - ts_log_diff)**2)
    plt.plot(ts_log_diff, label = 'Difference of Log Returns')    
    plt.plot(results_AR.fittedvalues, color = 'red', label = 'AR Model of Returns')
    plt.title('AR_Model RSS : %.6f'% RSS)
    plt.legend(loc = 'best')
    plt.show()
    
def MA_model(ts):
    
    log_ts = np.log(ts + 1)
    ts_log_diff = log_ts - log_ts.shift()
    ts_log_diff.dropna(inplace = True)
    
    model = ARIMA(log_ts, order = (0, 1, 1))
    results_MA = model.fit(disp = -1)
    RSS = sum((results_MA.fittedvalues - ts_log_diff)**2)
    plt.plot(ts_log_diff, label = 'Difference of Log Returns')    
    plt.plot(results_MA.fittedvalues, color = 'red', label = 'MA Model of Returns')
    plt.title('MA_Model RSS : %.6f'% RSS)
    plt.legend(loc = 'best')
    plt.show()
    
def ARIMA_model(ts):
    
    log_ts = np.log(ts + 1)
    ts_log_diff = log_ts - log_ts.shift()
    ts_log_diff.dropna(inplace = True)
    
    model = ARIMA(log_ts, order = (1, 1, 1))
    results_ARIMA = model.fit(disp = -1)    
    RSS = sum((results_ARIMA.fittedvalues - ts_log_diff)**2)
    plt.plot(ts_log_diff, label = 'Difference of Log Returns')    
    plt.plot(results_ARIMA.fittedvalues, color = 'red', label = 'ARIMA Model of Returns')
    plt.title('ARIMA_Model RSS : %.6f'% RSS)
    plt.legend(loc = 'best')
    plt.show()
    print (log_ts.head())
    
def final_model(ts):
    
    #Using ARIMA model to get modelled data as it gave lowest RSS but maximum likelihood optimization does not converge
    # So I wil use MA model 
    
    log_ts = np.log(ts + 1)
    ts_log_diff = log_ts - log_ts.shift()
    ts_log_diff.dropna(inplace = True)
    
    model = ARIMA(log_ts, order = (0, 1, 1))
    results_MA = model.fit(disp = -1)
    print (results_MA.summary())
    
    predictions_MA_diff = pd.Series(results_MA.fittedvalues, copy = True)
    
    #The way to convert the differencing to log scale is to add these differences consecutively to the base number.
    #An easy way to do it is to first determine the cumulative sum at index and then add it to the base number. 
    #predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    
    # Next we’ve to add them to base number
    # predictions_ARIMA_log = pd.Series(log_data.iloc[0], index=log_data.index)
    #base_data = 0 * predictions_ARIMA_diff_cumsum + float(log_ts.iloc[0])
    #predictions_ARIMA_log = base_data.add(predictions_ARIMA_diff.cumsum(), fill_value=0)
    
    #Last step is to take exponent and add values cumulatively from there
    #predictions_ARIMA = np.exp(predictions_ARIMA_log) - 1
    
    plt.plot(ts, label = 'Returns')
    plt.plot(predictions_MA_diff, label = 'MA model Return predictions')   
    RMS = sum((predictions_MA_diff - ts[1:])**2)
    plt.title('RMS : %.6f'% RMS)
    plt.legend()
    
    # plot residual errors
    residuals = pd.DataFrame(results_MA.resid)
    residuals.plot(title = 'Residual Plot')
    residuals.plot(kind='kde', title = 'Residual Distribution')
    
    print(residuals.describe())

    plt.show()
    
    """
                                     ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:                D.Close   No. Observations:                  998
    Model:                 ARIMA(0, 1, 1)   Log Likelihood                2609.144
    Method:                       css-mle   S.D. of innovations              0.018
    Date:                Sun, 04 Oct 2020   AIC                          -5212.288
    Time:                        15:40:34   BIC                          -5197.571
    Sample:                             1   HQIC                         -5206.694
                                                                                  
    =================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const          9.434e-07   1.94e-06      0.487      0.626   -2.85e-06    4.74e-06
    ma.L1.D.Close    -1.0000      0.003   -389.043      0.000      -1.005      -0.995
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    MA.1            1.0000           +0.0000j            1.0000            0.0000
    -----------------------------------------------------------------------------
                    0
    count  998.000000
    mean    -0.000016
    std      0.017693
    min     -0.161110
    25%     -0.006813
    50%     -0.000115
    75%      0.007837
    max      0.131444
"""