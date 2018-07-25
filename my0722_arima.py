import pandas
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy

df = pandas.read_csv('https://raw.githubusercontent.com/ywchiu/tibamedl/master/Data/2330.TW.csv')
df = df[~ df['Close'].isna()]

trainset = df.iloc[0:1551, :]
testset = df.iloc[1551:, :]

print('trainset.shape: ', trainset.shape)
print('testset.shape: ', testset.shape)


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


X = trainset.Close.values
differenced = difference(X, 365)

print('len(differenced): ', len(differenced))

model = ARIMA(differenced, order=(7, 0, 1))
model_fit = model.fit(disp=0)

start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)

print('len(forecast): ', len(forecast))

history = [x for x in X]
day = 1
for yhat in forecast:
    inverted = inverse_difference(history, yhat)
    print('Day %d: %f' % (day, inverted))
    history.append(inverted)
    day += 1
