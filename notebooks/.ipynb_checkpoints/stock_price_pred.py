import  pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials 
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import sys
from datetime import timedelta
import datetime





def load_data(start_date,end_date,ticker):
    """the functions read data from the yahoo website.
     and returns a dataframe and a list
     
     Input: start date: date format
            end date : date format
            ticker: name of the stock: string
     Output: Target variable:list
             Independent variables:dataframe"""
    df = yf.download(ticker,start_date,end_date,progress = False)
    X= df[['Open','High','Low','Close','Volume']]
    Y = df['Adj Close']
    return X,Y



def train_test_split(X,Y,end_date,max_date):
    
    """split the data to train and test sets
       input: X,Y
       output: X_train: dataframe
               Y_train: list
               X_test:dataframe
               Y_test:list"""

    X_train= X[:max_date]
    Y_train= Y[:max_date]
    X_test = X[max_date:end_date]
    Y_test = Y[max_date:end_date]
    #print(X_test)
    return X_train,Y_train,X_test,Y_test


def build_model():
    
    pipeline = Pipeline([
        ('clf', RandomForestRegressor())
    ]) 

    return pipeline



def evaluate_model(model,X_test,Y_test,ticker):
    #print(Y_test)
    Y_pred = model.predict(X_test)
    predicted = pd.DataFrame(Y_test)
    predicted['Adj Close Forecasted'] = Y_pred
    predicted['Ticker'] = ticker
    mape = abs(predicted['Adj Close']-predicted['Adj Close Forecasted'])/predicted['Adj Close']*100
    predicted['mape'] = mape
    mean_mape = predicted.mape.mean()
    mse= mean_squared_error(Y_test,Y_pred)
    print(predicted)
    print('the mean squared error:{}' .format(mse))
    print('the mean average percentage error:{}' .format(mean_mape))
    




def main():
    if len(sys.argv) == 5:
        
        start_date,end_date,max_date,ticker = sys.argv[1:]
        print('Loading data...\n    start_date: {} end_date:{} ticker: {}' .format(start_date,end_date,ticker))
        X, Y = load_data(start_date,end_date,ticker)
        print('split_data...\n    maximum training date:{}' .format(max_date))
        X_train,Y_train,X_test,Y_test =  train_test_split(X,Y,end_date,max_date)
        print('model training')
        model = build_model()
        print('model fitting')
        model.fit(X_train,Y_train)
        print('model evaluation')
        evaluate_model(model,X_test,Y_test,ticker)
    else:
        print('cannot run prediction')



if __name__ == '__main__':
    main()
