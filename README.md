# Backtest-ARIMA-Model-Startegy
Backtested trading strategy based on modelling stock returns based on Auto Regressive Integrated Moving Average model

Install
-------

This project uses [backtrader](https://www.backtrader.com/) and [pmdarima](https://pypi.org/project/pmdarima/). Go check them out if you don't have them locally installed.

```source-shell
$ pip install backtrader[plotting]
$ pip install pmdarima
```

Interpretation 
-------

Dive right into functions.py. I have added a lot of interpretations right after the various functions used in comments. This file starts with normal operations that are 
required to be done and checked when handling time series to understand it. I have used returns from msft stock to plot all the results (in the ```plots``` directory).

***Operations covered in functions.py*** - 

1.  one_period-multi_period-continuosly_compounded returns
2.  Plotted returns distribution histogram along with normal distribution histogram on same plot
3.  Performed Ljung-Box Test and plotted autocorrelation lags plot
4.  Plotted scatter plot between retruns and 1-day lag return
5.  Performed Augmented Dickey–Fuller test and plotted rolling mean, deviation and returns to check stationarity
6.  Log-Difference Returns taken to remove trend and increase stationarity of time series 
7.  Time series decomposed into residuals, trend and seasonality components and plotted
8.  Autocorrelation(ACF) and Partial Autocorrelation(PACF) plotted to find q and p resp. for the ARIMA model
9.  AR(1, 1, 0), MA(0, 1, 1), ARIMA(1, 1, 1) model fitted to logarithmic difference of returns. RSS values compared for each to find best fit

***Backtesting -***

First I have written a [custom indicator](https://www.backtrader.com/docu/inddev/) - *class ARIMA_ind* to generate a positive buying signal whenever the returns predicted by the 
ARIMA based model on the past window(20 or 30 days) of current time is positive and go short if predicted results are negative. 

I have made 2 indicators - 

1.  ARIMA(0, 1, 1) with lag_window = 20 days
2.  This one selects the best ARIMA model for the lag period on its own. I have used [pmdarima](https://pypi.org/project/pmdarima/) library for this.

The 2 generated plots are in the ```backtest_plots``` directory

Contributing
------------

Feel free to dive in! Open an issue or submit PRs.
