import pandas as pd 
import quandl
import math , datetime
import numpy as np
from sklearn import preprocessing  , svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt
# pickling is serialization of any object , dictionary , ...
import pickle
import os.path

style.use('ggplot')

quandl.ApiConfig.api_key = "9x2youC9-Dw3otSMtzJ4"

df = quandl.get('WIKI/GOOGL')


# using splitted stock shares (adjusted)
df = df[['Adj. Open' , 'Adj. High' , 'Adj. Low' , 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100

df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


forecast_col = 'Adj. Close'
# replace NAN with -99999
df.fillna(-99999 , inplace = True)
print(df.head())
# 10 percent of data to forecast
forecast_out = int(math.ceil(0.1*len(df)))
# my guess : we shift one column , some data is NAN now 
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head())
# we get rid of the rows with NAN values just created

# label is Y literaly 
X = np.array(df.drop(['label' , 'Adj. Close'] , 1))


# feature scaling 
X = preprocessing.scale(X)
# we should scale every new data and scale them all along ! 

X_lately = X[-forecast_out:]
X = X[:-forecast_out]



df.dropna(inplace=True)
Y = np.array(df['label'])
# Y_lately = Y[-forecast_out:]
# Y = Y[:-forecast_out]

print("First Step Completed")

# train and test must be seperate 
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

if os.path.isfile('linearregression.pickle'):

    print("PreExisting Classifier")
    pickle_in = open('linearregression.pickle' , 'rb')
    clf = pickle.load(pickle_in)

else:
    print("Creating Classifier")
    #linear regression in 2 threads (-1 if want maximum)
    clf = LinearRegression(n_jobs=2)
    # svm no kernel
    # clf = svm.SVR()
    # svm with polynomial kernel
    # clf = svm.SVR(kernel='poly')

    # train
    clf.fit(X_train , Y_train)
    # we can save the classifier
    with open('linearregression.pickle' , 'wb') as f:
        pickle.dump(clf , f)


# test (by square error)
accuracy = clf.score(X_test,Y_test)
print(accuracy)

# svm without kerlen: 0.776
# svm with kernel : 0.659
# linear regression : 0.979 (Great)


forecast_set = clf.predict(X_lately)

#print(forecast_set, accuracy , forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
#print(df.iloc[-1].name)
last_unix = last_date.timestamp()
one_day = 86000
next_unix = last_unix + one_day 


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
#print(df.tail())

# df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()