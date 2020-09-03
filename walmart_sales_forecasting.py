import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
from statsmodels.tsa.stattools import kpss
from sklearn import preprocessing, metrics
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import RepeatVector,TimeDistributed
import tqdm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
import pickle
import regex as re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
import gc
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

warnings.filterwarnings("ignore")

df_calendar = pd.read_csv("/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/calendar.csv")
df_sell_prices = pd.read_csv("/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/sell_prices.csv")
df_sales_train = pd.read_csv("/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/sales_train_validation.csv")

'''
df_calendar = pd.DataFrame(df_calendar)
df_sell_prices = pd.DataFrame(df_sell_prices)
df_sales_train = pd.DataFrame(df_sales_train)

'''

print(df_calendar.shape)
print(df_calendar.head(3))

print(df_sell_prices.shape)
print(df_sell_prices.head(3))

print(df_sales_train.shape)
print(df_sales_train.head(3))

# Saving a Dataframe
'''
df_calendar.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_calendar.pkl')
df_calendar_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_calendar.pkl')
'''

# Convert the "d_1" ...... "d_1913" to datetime
date_1 = df_calendar['date'][0:len(df_sales_train.loc[0, 'd_1':])]
date_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date_1]
print(type(date_list[0]))


df_sales_train['item_store_id'] = df_sales_train.apply(lambda x: x['id'].replace('_validation', ''), axis=1)
df_sales = df_sales_train.loc[:, 'd_1':'d_1913'].T
df_sales.columns = df_sales_train['item_store_id'].values
df_sales = df_sales.set_index([date_list])
print(df_sales.iloc[:,0])
print(df_sales.head(3))


df_sales_preprocessed_lstm = df_sales
df_sales_preprocessed_lstm.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_sales_preprocessed_lstm.pkl')



# Plot an arbitrary column of the df_sales dataset
my_dpi = 300
base_path = os.path.abspath('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy')

def plot_ts(n, size_x_inch, size_y_inch, color, line_width, title):
    y = df_sales.iloc[:, n]
    y.index = pd.to_datetime(y.index)
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y.index, y.values, linewidth=line_width, color=color)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.savefig(os.path.join(base_path, title + '.png'), dpi=my_dpi)
    plt.show()


def plot_ts_diff(n, size_x_inch, size_y_inch, color, line_width, title):
    y = df_sales.iloc[:, n]
    y.index = pd.to_datetime(y.index)
    y_1 = y[0:len(y)-1]
    y_2 = y[1:]
    y_diff = y_2.values - y_1.values
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y_1.index, y_diff, linewidth=line_width, color=color)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.savefig(os.path.join(base_path, title + '.png'), dpi=my_dpi)
    plt.show()

### Get the ACF and PACF of a time series and the first order difference of time series
test = df_sales.iloc[:, 100]
test.index = pd.to_datetime(test.index)
test_1 = test[0:len(test) - 1]
test_2 = test[1:]
test_diff = test_2.values - test_1.values

fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(test_diff, ax=ax[0], lags=200)
ax[1] = plot_pacf(test_diff, ax=ax[1], lags=200)

fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(test.values, ax=ax[0], lags=200)
ax[1] = plot_pacf(test.values, ax=ax[1], lags=200)



print(" > Is the data stationary ?")
dftest = kpss(test_diff, 'c')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[3].items():
    print("\t{}: {}".format(k, v))

print(f'Result: The series is {"not " if dftest[1] < 0.05 else ""}stationary')



# Calculate the RMSSE
def RMSSE(y, y_hat, n, h):
    numerator = 0
    dinominator = 0
    for i in range(1, n, 1):
        dinominator += (y[i]-y[i-1])**2
    for j in range(n, n+h, 1):
        numerator += (y[j]-y_hat[j])**2
    return ((n-1)*numerator)/(h*dinominator)


def plot_ts_snippet(n1, n2, n3, size_x_inch, size_y_inch, color, line_width):
    y = df_sales.iloc[:, n1]
    y_1 = y.iloc[n2*n3:n2*(n3+1)]
    y_1.index = pd.to_datetime(y_1.index)
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y_1.index, y_1.values, linewidth=line_width, color=color, label='Original data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()


def plot_ts_sma(n1, n2, n3, n4, size_x_inch, size_y_inch):
    y = df_sales.iloc[:, n1]
    y_1 = y.iloc[n2*n3:n2*(n3+1)]
    y_1.index = pd.to_datetime(y_1.index)
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y_1.index, y_1.values, linewidth=1, color='r', label='Original data')
    rolling_mean = y_1.rolling(window=n4).mean()
    rolling_mean.index = pd.to_datetime(rolling_mean.index)
    plt.plot(rolling_mean.index, rolling_mean.values, linewidth=1, color='b', label='SMA')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()




# Exponential moving average (EMA)/Exponential smoothing
# n4 is the moving average factor
def plot_ts_ema(n1, n2, n3, n4, size_x_inch, size_y_inch):
    y = df_sales.iloc[:, n1]
    y_1 = y.iloc[n2*n3:n2*(n3+1)]
    y_1.index = pd.to_datetime(y_1.index)
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y_1.index, y_1.values, linewidth=1, color='r', label='Original data')
    exp = y_1.ewm(span=n4, adjust=False).mean()
    exp.index = pd.to_datetime(exp.index)
    plt.plot(exp.index, exp.values, linewidth=1, color='b', label='EMA')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()


def plot_ts_holt(n1, n2, n3, smooth_level, smooth_slope, size_x_inch, size_y_inch):
    y = df_sales.iloc[:, n1]
    y_1 = y.iloc[n2*n3:n2*(n3+1)]
    y_1.index = pd.to_datetime(y_1.index)
    model = Holt(np.asarray(y_1))
    fit = model.fit(smoothing_level=smooth_level, smoothing_slope=smooth_slope)
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y_1.index, y_1.values, linewidth=1, color='r', label='Original data')
    plt.plot(y_1.index, fit.fittedvalues, linewidth=1, color='b', label='Double exponential smoothing')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()



def plot_ts_lowess(n1, n2, n3, size_x_inch, size_y_inch, filter_frac):
    y = df_sales.iloc[:, n1]
    y_1 = y.iloc[n2 * n3:n2 * (n3 + 1)]
    y_1.index = pd.to_datetime(y_1.index)
    filter_lowess = pd.DataFrame(lowess(y_1.values, np.arange(len(y_1.values)), is_sorted=True, frac=filter_frac, it=0))
    lowess_x = y_1.index
    lowess_y = filter_lowess[1]
    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(lowess_x, lowess_y, linewidth=1, color='b', label='lowess')
    plt.plot(y_1.index, y_1.values, linewidth=1, color='r', label='Original data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()



def plot_ts_standard_gaussian_kernel(n1, n2, n3, size_x_inch, size_y_inch, h):
    y = df_sales.iloc[:, n1]
    y_1 = y.iloc[n2 * n3:n2 * (n3 + 1)]
    y_1.index = pd.to_datetime(y_1.index)
    position = np.arange(len(y_1.values))
    m = []
    z = np.array(y_1.values)

    for i in range(len(position)):
        kernel_at_pos = np.exp(-((position-i)/h)**2/2)
        kernel_at_pos = kernel_at_pos/sum(kernel_at_pos)
        m.append(sum(kernel_at_pos*z))

    fig = plt.figure()
    fig.set_size_inches(size_x_inch, size_y_inch)
    plt.plot(y_1.index, m, linewidth=1, color='b', label='Standard gaussian kernel')
    plt.plot(y_1.index, y_1.values, linewidth=1, color='r', label='Original data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

################################################################################################################
# Classical and statistical method
# ARIMA
test = df_sales.iloc[:, 100]
test = test.astype('float64')


model_010 = ARIMA(test, (0, 1, 0))
res_010 = model_010.fit()
print(res_010.summary())


model_011 = ARIMA(test, (0, 1, 1))
res_011 = model_011.fit()
print(res_011.summary())


model_110 = ARIMA(test, (1, 1, 0))
res_110 = model_110.fit()
print(res_110.summary())


model_101 = ARIMA(test, (1, 0, 1))
res_101 = model_101.fit()
print(res_101.summary())


# Check if the redisual is stabilized around 0
# Calculate the variances of the residuals
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(1, 4))
ax[0].plot(res_010.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(res_010.resid.values)))
ax[0].hlines(0, xmin=0, xmax=1913, color='r')
ax[0].set_title("ARIMA(0,1,0)")
ax[0].legend()

ax[1].plot(res_011.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(res_011.resid.values)))
ax[1].hlines(0, xmin=0, xmax=1913, color='r')
ax[1].set_title("ARIMA(0,1,1)")
ax[1].legend()

ax[2].plot(res_110.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(res_110.resid.values)))
ax[2].hlines(0, xmin=0, xmax=1913, color='r')
ax[2].set_title("ARIMA(1,1,0)")
ax[2].legend()

ax[3].plot(res_101.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(res_101.resid.values)))
ax[3].hlines(0, xmin=0, xmax=1913, color='r')
ax[3].set_title("ARIMA(1,0,1)")
ax[3].legend()


df_arima = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_sales_preprocessed_lstm.pkl')
df_arima_tx = df_arima.filter(like='TX', axis = 1)
df_arima_tx_FOODS_1 = df_arima_tx.filter(like='FOODS_1', axis = 1)
print(df_arima_tx_FOODS_1.shape)

# n is a specific column of the df
def arima_time_series(n, arima_n_1, arima_n_2, arima_n_3):
    X = df_arima_tx_FOODS_1.iloc[:,n]
    train_1, test_1 = X[0:len(X) - 28], X[len(X) - 28:len(X)]
    history = [x for x in train_1]
    predictions = list()

    for t in range(len(test_1)):
        model = ARIMA(history, order=(arima_n_1, arima_n_2, arima_n_3))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(round(yhat[0]))
        obs = test_1[t]
        history.append(yhat[0])
        print('predicted=%f, expected=%f' % (round(yhat[0]), obs))

    arima_true = test_1.values
    arima_pred = np.array(predictions)
    return (arima_true, arima_pred)

##### Arima(1,1,1)
prediction_result_arima_1_1_1 = []
for i in range(df_arima_tx_FOODS_1.shape[1]):
    prediction_result_arima_1_1_1.append(arima_time_series(i, 1, 1, 1))

for i in range(582,df_arima_tx_FOODS_1.shape[1],1):
    prediction_result_arima_1_1_1.append(arima_time_series(i, 1, 1, 1))

y_true_all_1_1_1 = np.array([])
y_pred_all_1_1_1 = np.array([])

for result in prediction_result_arima_1_1_1:
    y_true_all_1_1_1 = np.concatenate([y_true_all_1_1_1, result[0]])
    y_pred_all_1_1_1 = np.concatenate([y_pred_all_1_1_1, result[1]])

from sklearn.metrics import mean_squared_error
from math import sqrt
rms_1_1_1 = sqrt(mean_squared_error(y_true_all_1_1_1, y_pred_all_1_1_1))

np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_true_all_1_1_1.npy', y_true_all_1_1_1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_pred_all_1_1_1.npy', y_pred_all_1_1_1)

y_true_all_1_1_1_load = np.load('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_true_all_1_1_1.npy')
y_pred_all_1_1_1_load = np.load('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_pred_all_1_1_1.npy')


rms_1_1_1 = np.sqrt(mean_squared_error(y_true_all_1_1_1_load, y_pred_all_1_1_1_load))



# Arima(3,1,1)
prediction_result_arima_3_1_1 = []
for i in range(df_arima_tx_FOODS_1.shape[1]):
    prediction_result_arima_3_1_1.append(arima_time_series(i, 3, 1, 1))

for i in range(564,df_arima_tx_FOODS_1.shape[1],1):
    prediction_result_arima_3_1_1.append(arima_time_series(i, 3, 1, 1))

y_true_all_3_1_1 = np.array([])
y_pred_all_3_1_1 = np.array([])

for result in prediction_result_arima_3_1_1:
    y_true_all_3_1_1 = np.concatenate([y_true_all_3_1_1, result[0]])
    y_pred_all_3_1_1 = np.concatenate([y_pred_all_3_1_1, result[1]])

from sklearn.metrics import mean_squared_error
from math import sqrt
rms_3_1_1 = sqrt(mean_squared_error(y_true_all_3_1_1, y_pred_all_3_1_1))

np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_true_all_3_1_1.npy', y_true_all_3_1_1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_pred_all_3_1_1.npy', y_pred_all_3_1_1)




# Arima(5,1,1)
prediction_result_arima_5_1_1 = []
for i in range(df_arima_tx_FOODS_1.shape[1]):
    prediction_result_arima_5_1_1.append(arima_time_series(i, 5, 1, 1))

y_true_all_5_1_1 = np.array([])
y_pred_all_5_1_1 = np.array([])

for result in prediction_result_arima_5_1_1:
    y_true_all_5_1_1 = np.concatenate([y_true_all_5_1_1, result[0]])
    y_pred_all_5_1_1 = np.concatenate([y_pred_all_5_1_1, result[1]])

from sklearn.metrics import mean_squared_error
from math import sqrt
rms_5_1_1 = sqrt(mean_squared_error(y_true_all_5_1_1, y_pred_all_5_1_1))

np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_true_all_5_1_1.npy', y_true_all_5_1_1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_pred_all_5_1_1.npy', y_pred_all_5_1_1)

y_true_all_5_1_1_load = np.load('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_true_all_5_1_1.npy')
y_pred_all_5_1_1_load = np.load('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/y_pred_all_5_1_1.npy')


rms_5_1_1 = np.sqrt(mean_squared_error(y_true_all_5_1_1_load, y_pred_all_5_1_1_load))
################################################################################################################





# Machine learning method
################################################################################################################
# Random forest regression (RFR)

df_cal = pd.read_csv('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/calendar.csv', parse_dates = ['date'])
df_sales_train = pd.read_csv('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/sales_train_validation.csv')
df_prices = pd.read_csv('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/sell_prices.csv')

list_id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_d_cols = df_sales_train.drop(list_id_vars, axis=1)

df_d_cols_train = df_d_cols.loc[:,'d_1':'d_1885']
df_d_cols_test = df_d_cols.loc[:, 'd_1886':]

df_sales_train_train = df_sales_train.loc[:, 'id':'d_1885']

df_sales_train_test_1 = df_sales_train.loc[:,'id':'state_id']
df_sales_train_test_2 = df_sales_train.loc[:,'d_1886':]
df_sales_train_test = pd.concat([df_sales_train_test_1, df_sales_train_test_2], axis=1)

df_melted_sales_train = df_sales_train_train.melt(id_vars = list_id_vars, value_vars = df_d_cols_train.columns, var_name = 'd', value_name = 'sales')
df_melted_sales_test = df_sales_train_test.melt(id_vars = list_id_vars, value_vars = df_d_cols_test.columns, var_name = 'd', value_name = 'sales')


df_melted_sales_train.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace = True)
df_melted_sales_test.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace = True)

df_train = df_melted_sales_train.merge(df_cal, left_on='d', right_on='d', how='left')
df_test = df_melted_sales_test.merge(df_cal, left_on='d', right_on='d', how='left')


# Save file
df_train.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train.pkl')
df_test.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test.pkl')
# Import file
df_train_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train.pkl')
df_test_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test.pkl')


df_prices['id'] = df_prices['item_id'] +'_' + df_prices['store_id']
df_prices.drop(['item_id', 'store_id'], axis=1, inplace = True)
df_prices.head()

df_train_loaded['id_for_price'] = df_train_loaded['id'].str.replace('_validation','')
df_test_loaded['id_for_price'] = df_test_loaded['id'].str.replace('_validation','')

df_train_loaded = pd.merge(df_train_loaded, df_prices,  how='left', left_on=['id_for_price', 'wm_yr_wk'],right_on = ['id', 'wm_yr_wk'])
df_test_loaded = pd.merge(df_test_loaded, df_prices,  how='left', left_on=['id_for_price', 'wm_yr_wk'],right_on = ['id', 'wm_yr_wk'])

df_train_loaded.drop(['id_for_price', 'id_y'], axis=1, inplace = True)
df_train_loaded.rename(columns = {"id_x":"id"}, inplace = True)
df_test_loaded.drop(['id_for_price', 'id_y'], axis=1, inplace = True)
df_test_loaded.rename(columns = {"id_x":"id"}, inplace = True)


df_train_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_new.pkl')
df_test_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_new.pkl')


df_test_new_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_new.pkl')
df_test_tx = df_test_new_loaded[df_test_new_loaded['id'].str.contains("TX")]
df_test_tx.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx.pkl')


df_train_new_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_new.pkl')
df_train_tx = df_train_new_loaded[df_train_new_loaded['id'].str.contains("TX")]
df_train_tx.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx.pkl')


df_train_tx_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx.pkl')
df_test_tx_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx.pkl')


df_test_tx_FOODS_1 = df_test_tx_loaded[df_test_tx_loaded['id'].str.contains("FOODS_1")]
df_test_tx_FOODS_1.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1.pkl')


df_train_tx_FOODS_1 = df_train_tx_loaded[df_train_tx_loaded['id'].str.contains("FOODS_1")]
df_train_tx_FOODS_1.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1.pkl')


df_train_tx_FOODS_1_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1.pkl')
df_test_tx_FOODS_1_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1.pkl')


df_train_tx_FOODS_1_loaded.drop(['snap_CA', 'snap_WI'], axis=1, inplace = True)
df_test_tx_FOODS_1_loaded.drop(['snap_CA', 'snap_WI'], axis=1, inplace = True)


df_train_tx_FOODS_1_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1_new.pkl')
df_test_tx_FOODS_1_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1_new.pkl')



##### staring from here
df_train_tx_FOODS_1_new = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1_new.pkl')
df_test_tx_FOODS_1_new = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1_new.pkl')
df_train_tx_FOODS_1_new.columns[df_train_tx_FOODS_1_new.isnull().any()]
df_test_tx_FOODS_1_new.columns[df_test_tx_FOODS_1_new.isnull().any()]

df_train_tx_FOODS_1_new_is_null = df_train_tx_FOODS_1_new[df_train_tx_FOODS_1_new['sell_price'].isnull()]
df_train_tx_FOODS_1_new_price_not_null = df_train_tx_FOODS_1_new[df_train_tx_FOODS_1_new['sell_price'].notnull()]

imputed_values = {'event_name_1' :'None', 'event_type_1' :'None', 'event_name_2' :'None', 'event_type_2' :'None'}

df_train_tx_FOODS_1_new_price_not_null.fillna(value = imputed_values, inplace = True)
df_test_tx_FOODS_1_new.fillna(value = imputed_values, inplace = True)
df_train_tx_FOODS_1_new_price_not_null.columns[df_train_tx_FOODS_1_new_price_not_null.isnull().any()]
df_test_tx_FOODS_1_new.columns[df_test_tx_FOODS_1_new.isnull().any()]

# list all non-numeric data
df_train_tx_FOODS_1_new_price_not_null.select_dtypes(include = 'object').columns
df_test_tx_FOODS_1_new.select_dtypes(include = 'object').columns

drop_columns = ['weekday','d']

df_train_tx_FOODS_1_new_price_not_null.drop(drop_columns, axis=1, inplace = True)
df_test_tx_FOODS_1_new.drop(drop_columns, axis=1, inplace = True)

# These are functions from fast.ai that convert objects to categories and keeps those categories consistent across training and test
train_cats(df_train_tx_FOODS_1_new_price_not_null)
apply_cats(df_test_tx_FOODS_1_new, df_train_tx_FOODS_1_new_price_not_null)

cat_cols = df_train_tx_FOODS_1_new_price_not_null.select_dtypes(include = 'category').columns

for i in cat_cols:
    df_train_tx_FOODS_1_new_price_not_null['cat_'+i] = df_train_tx_FOODS_1_new_price_not_null[i].cat.codes
    df_test_tx_FOODS_1_new['cat_'+i] = df_test_tx_FOODS_1_new[i].cat.codes

df_train_tx_FOODS_1_new_price_not_null.drop(cat_cols, axis = 1, inplace = True)
df_test_tx_FOODS_1_new.drop(cat_cols, axis = 1, inplace = True)

df_train_tx_FOODS_1_new_price_not_null.select_dtypes(include = 'category').columns
df_test_tx_FOODS_1_new.select_dtypes(include = 'category').columns

df_train_tx_FOODS_1_new_price_not_null.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1_updated.pkl')
df_test_tx_FOODS_1_new.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1_updated.pkl')



###### The final data preparation
df_train_tx_FOODS_1_updated = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1_updated.pkl')
df_test_tx_FOODS_1_updated = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1_updated.pkl')

need_to_drop = ['year', 'wday', 'month']

df_train_tx_FOODS_1_updated.drop(need_to_drop, axis=1, inplace = True)
df_test_tx_FOODS_1_updated.drop(need_to_drop, axis=1, inplace = True)

date_cols = df_train_tx_FOODS_1_updated.select_dtypes(include = 'datetime64').columns
for i in date_cols:
    add_datepart(df_train_tx_FOODS_1_updated, i)
    add_datepart(df_test_tx_FOODS_1_updated, i)

df_train_tx_FOODS_1_updated.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1_final.pkl')
df_test_tx_FOODS_1_updated.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1_final.pkl')

df_train_tx_FOODS_1_final = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_train_tx_FOODS_1_final.pkl')
df_test_tx_FOODS_1_final = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/df_test_tx_FOODS_1_final.pkl')


## Training
y_train = df_train_tx_FOODS_1_final['sales']
y_test = df_test_tx_FOODS_1_final['sales']

df_train_tx_FOODS_1_final.pop('sales')
df_test_tx_FOODS_1_final.pop('sales')

model = RandomForestRegressor(n_jobs = -1)
model.fit(df_train_tx_FOODS_1_final, y_train)

Model_file = '/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/random_forest_model.pkl'
pickle.dump(model, open(Model_file, 'wb'))
loaded_model = pickle.load(open(Model_file, 'rb'))

prediction = loaded_model.predict(df_test_tx_FOODS_1_final)
pred_round = np.round(prediction)
pred_round.astype(int)
pred = pred_round
true = np.array(y_test)

plt.plot(true[:100])
plt.plot(pred[:100])
plt.show()

rms_rf = sqrt(mean_squared_error(true, pred))
################################################################################################################






################################################################################################################
# Recurrent neural networks (RNNs) - LSTM
df_lstm_new = df_sales_preprocessed_lstm
df_lstm_tx = df_lstm_new.filter(like='TX', axis = 1)
df_lstm_tx_FOODS_1 = df_lstm_tx.filter(like='FOODS_1', axis = 1)
print(df_lstm_tx_FOODS_1.shape)
# The shape is (1913, 648)

def sliding_window(training_data, seq_length):
    x = []
    y = []
    for i in range(len(training_data)-1-seq_length):
        x_1 = training_data[i:(i+seq_length)]
        y_1 = training_data[i+seq_length]
        x.append(x_1)
        y.append(y_1)
    return np.array(x),np.array(y)

# n: index of column of the dataframe
def time_series_lstm(n, window_size):
    df_lstm = df_lstm_tx_FOODS_1.iloc[:,n]
    df_lstm.index = pd.to_datetime(df_lstm.index)
    data = np.array(df_lstm)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_lstm_normalized = scaler.fit_transform(data.reshape(-1, 1))
    df_lstm_normalized_training = df_lstm_normalized[0:len(df_lstm_normalized) - window_size]
    df_lstm_normalized_testing = df_lstm_normalized[len(df_lstm_normalized) - window_size:]
    df_lstm_testing = df_lstm[len(df_lstm_normalized) - window_size:]
    X_train, y = sliding_window(df_lstm_normalized_training, window_size)
    X_train = np.array(X_train)
    n_features = 1
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    y = y.reshape((y.shape[0], y.shape[1], 1))

    y_1 = []
    for i in range(len(y)):
        y_1.append(y[i][0])
    y_1 = np.array(y_1)

    np.random.seed(100)
    if __name__ == '__main__':
        model = Sequential()
        model.add(LSTM(units=28, activation='relu', return_sequences=True, input_shape=(28, 1)))
        model.add(LSTM(units=28, activation='relu', return_sequences=True, input_shape=(28, 1)))
        model.add(LSTM(units=28, activation='relu', return_sequences=False, input_shape=(28, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_1, epochs=10, batch_size=50)

    X_input = X_train[-1]
    ## Predict the next 28 days
    y_hat_all = list()
    y_hat_all_1 = list()
    for i in range(28):
        X_input = X_input.reshape(1, 28, 1)
        y_hat = model.predict(X_input, verbose=0)
        y_hat_all_1.append(y_hat)
        y_hat_all.append(y_hat[0][0])
        X_input = np.append(X_input[0][1:], [y_hat[0][0]])

    y_regular_scale = []
    for data in y_hat_all_1:
        y_regular_scale.append(scaler.inverse_transform(data))

    y_hat_modified = []
    for i in range(len(y_regular_scale)):
        y_hat_modified.append(y_regular_scale[i][0][0])

    y_hat_modified_1 = []
    for num in y_hat_modified:
        y_hat_modified_1.append(round(num))

    y_true = df_lstm_testing.values
    y_pred = y_hat_modified_1
    return (y_true, y_pred)

prediction_result = []
for i in range(df_lstm_tx_FOODS_1.shape[1]):
    prediction_result.append(time_series_lstm(i,28))

lstm_prediction_result = np.array(prediction_result)
np.save('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/lstm_prediction_result.npy', lstm_prediction_result)
lstm_prediction_result_loaded = np.load('/Users/ganfeng/Documents/Data Science/Projects/m5-forecasting-accuracy/data/lstm_prediction_result.npy')

y_true_all = np.array([])
y_pred_all = np.array([])

for result in lstm_prediction_result_loaded:
    y_true_all = np.concatenate([y_true_all, result[0]])
    y_pred_all = np.concatenate([y_pred_all, result[1]])

import matplotlib.pyplot as plt
plt.plot(y_true_all[:2000],'b')
plt.plot(y_pred_all[:2000],'r')
plt.show()

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_true_all, y_pred_all))
################################################################################################################
