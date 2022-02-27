
from training import create_time_features

import pickle
import pandas as pd
import sys
import argparse



def load_model(model_path):
    """ the functions loads the model that have been trained and saved.
    INPUT: the model path
    Output: A dictionary of the model
    """
    filename = model_path 
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model



def predict (symbol_list,models,start_date,end_date):

    """the functions forecast the prices of the stock for the list 
    of symbols provided and for the time period specified

    Input: model object: pickel file
         : List of symbols to predict the prices:list
         : Start date- date from the point you need forecast: data formart
         : End date - last date to forecast: date format

    Output: data frame with the forecasted price at a day level for the symbols specified:data frame

    """

    forecasted_df = []
    for symbol in symbol_list:
    
        df = pd.DataFrame(pd.date_range(start_date,end_date,freq='d'))
        #print(df)
        df.columns = ['ds']

        df = create_time_features(df)
        features = df.drop(columns= 'ds')
        
        model = models[symbol]
        forecasted = model.predict(features)
        df['forecasted_price'] = forecasted
        df['Symbol']= symbol
        df = df[['ds','Symbol','forecasted_price']]
        forecasted_df.append(df)
    forecasted_df = pd.concat(forecasted_df)
    return forecasted_df



def main(start_date,end_date,model_path,symbol_list):
    print('loading the models')
    models = load_model(model_path)
    print('forecasting the stock prices....\n  start_date:{} end_date:{}'.format(start_date,end_date))
    forecasted_df = predict(symbol_list, models, start_date, end_date)
    print(forecasted_df)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date',help = 'minimum date for you prediction:date format')
    parser.add_argument('end_date',help = 'maximum date for you prediction:data format')
    parser.add_argument('model_path',help= 'the path of the saved models')
    parser.add_argument('symbols_list',nargs="+",help = 'list of the symbols to predict on')
    args = parser.parse_args()
    main(args.start_date,args.end_date,args.model_path,args.symbols_list)