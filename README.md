final_project
==============================

udacity nano degree final project

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.

    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.

    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py


### Project Description
Investment in stock trading is referred to as passive mode of investment. Though passive, the risk in stock trading is very high. To hedge such risk, one needs to have a good understanding of the stock prospects. This can be done using the company’s information that is available to the public. Such information includes the historical trading prices. Historical trends though good may not be enough to decide whether to invest or not. A future outlook of how the prices will trend is required. To a achieve that, a clear understanding of the underlying patterns is required and with a use of a machine learning models we can forecast the future stock prices for a given period within an acceptable error rate.

In this project, using the historical data a model for every specified stock is trained and used to forecast the future prices.

### Instructions

The project will use a simple script. 
-- Training Script: Does the Model training and saves the model, it also returns the error rate for the predicted days
to run the training script you need to parse the following arguments
1. test size- an integer. The size of the data that will be used to evalute the trained model.
2. Start Date: date format. Specify the minimum date for the training dataset
3. End Date: date format. Specify the maximum date for the training data set
4. A folder path to save the trained models
5. Symbols List: A list of stocks intedend to train a model on

- To run the training script `python training.py test_size start_date end_date folder_path list_of_stocks`

NB:for the list just simply the symbols of the stock comma separated.e.g AAPL,GOOGL


-- Prediction script: Takes the saved model and forecast the price of the stocks provided 
to run the training script you need to parse the following arguments
1. start_date: date format. Minimum date for the forecast
2. end_date: date format. Maximum date for the forecast
3. Model Path: the path where the trained models are saved
4. Symbols List: A list of stocks to forecast on

- To run the prediction script `python predict.py start_date end_date folder_path list_of_stocks`

NB:For you to forecast a given stock, need to have trained a model for that specific stock


### Git Hub Repo 


[githup repo](https://github.com/dgitahi/nano-degree-final_project)

For more details on the analysis you can refer to the [medium story](https://medium.com/@gitahidave/stock-prediction-analysis-296e001e678b)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
