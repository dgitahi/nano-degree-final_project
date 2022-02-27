

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from pandas_datareader import data
from sklearn.ensemble import RandomForestRegressor
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")


def load_data(symbols, start_date, end_date):

    """
    INPUT:
    tickers : list containing the tickers of the stocks whose prices will be predicted
    start_date : initial date to gather data
    end_data : final date to gather data
    OUTPUT:
    prices_base : dataframe containing the adjusted closing price for the stocks
                on the desired time frame
    """
    df = data.DataReader(
        symbols,
        'yahoo',
        start_date,
        end_date)

    df = pd.DataFrame(df)
    df_base = df['Adj Close']

    try: 
        
        df_prices = df_base.stack()
        df_prices = pd.DataFrame(df_prices)
        df_prices.columns = ['y']
    
    except:
        df = pd.DataFrame(df_base)
        df['Symbols'] = symbols
        df.rename(columns= {'Adj Close':'y'},inplace = True)
        df_prices = df
    
    #print(df_prices)

    df_prices.reset_index(inplace=True)
    df_prices = df_prices.sort_values(by = ['Symbols','Date'])
    df_prices.rename(columns = {'Date':'ds'},inplace = True)

    return df_prices



def create_time_features(df):
    """the function create time features using the date column
    and returns a dataframe
    
    Input:  dataframe
    Output: dataframe that has all the time features
    """
        
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['sin_day'] = np.sin(df['dayofyear'])
    df['cos_day'] = np.cos(df['dayofyear'])
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.weekofyear
    df=df.sort_values('ds')
    return df



def split_data(test_size, df):

    """
    Split data into training and testing sets

    INPUT:
    test_size - size of testing data in number- number of days to forecast ahead
    df - dataframe
    OUTPUT:
    X_train -dataframe with the training features
    y_train- list of the response variable
    X_test - dataframe with the test features
    y_test- list- response variable
    """
    
#     test_size = test_size
#     training_size = 1 - test_size

#     test_num = int(test_size * len(df))
#     train_num = int(training_size * len(df))
    df =df.set_index('ds')
    train_size = len(df)-int(test_size)
    
    train = df[:train_size].drop(columns= ['Symbols'])
    test = df[train_size:].drop(columns= ['Symbols'])
    
    X_train=train.drop(columns='y')
    X_test =test.drop(columns='y')
    y_train = train['y']
    y_test = test['y']
    return X_train,y_train,X_test,y_test



def build_model():

    pipeline = Pipeline([
        ('clf', RandomForestRegressor(n_estimators=120))
    ]) 

    return pipeline






def evaluate_model(model,X_test,Y_test):
    """ the functions evaluates the model performance and returns a
        Mean Average Error for the specified period
        
        Input: trained model, 
            Dataframe with the test features 
            a list of the response variables
        Output:Data Frame with the predictions and error at the daily level
            Mean average Percentage Error for the Period
        
        """
    Y_pred = model.predict(X_test)
    predicted = pd.DataFrame(Y_test)
    predicted['y_Forecasted'] = Y_pred
    mape = abs(predicted['y']-predicted['y_Forecasted'])/predicted['y']*100
    predicted['mape'] = mape
    mean_mape = predicted.mape.mean()
    return predicted,mean_mape
    

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)




    

def main(test_size,start_date,end_date,model_path,symbol_list):

   

    #if len(sys.argv) == 6:
    print('Loading data ...\n    start_date: {} end_date:{} symbols: {}' .format(start_date,end_date,symbol_list))
    df = load_data(symbol_list,start_date,end_date)
    print('creating time features')
    df = create_time_features(df)

    error= []
    predicted_df = []

    symbols = df.Symbols.unique()
    models = {}
    for symbol in symbols:
        print(symbol)
        df_train = df[df.Symbols==symbol]
        
        print('model building')
        model = build_model()
        models[symbol] = model
        print('split_data...\n    test_size:{}' .format(test_size))

        X_train,y_train,X_test,y_test = split_data(test_size,df_train)

        model.fit(X_train,y_train)
        predicted,mean_mape = evaluate_model(model,X_test,y_test)
        predicted['Symbol'] = symbol

        error.append(mean_mape)
        predicted_df.append(predicted)
    error_df = pd.DataFrame({'Symbol':symbols,'Error':error})
    predicted_df = pd.concat(predicted_df)
    print(error_df)
    print(predicted_df)

    print('saving the model')
    
    save_model(models, model_path)

    print('Trained model saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_size', help='the size of the test group as an integer')
    parser.add_argument('start_date', default= '2020-01-04',help='the date from when the training set will start: date format')
    parser.add_argument('end_date', help='the last date of your training set:date format')
    parser.add_argument('model_path', help='the path where the models will be saved')
    parser.add_argument('symbol_list', nargs="+",help='Alist of the names of the stocks')
    args = parser.parse_args()
    main(args.test_size,args.start_date,args.end_date,args.model_path,args.symbol_list)
    

 