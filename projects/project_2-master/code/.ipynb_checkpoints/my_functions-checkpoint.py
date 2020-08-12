#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain # Pradeep Elance https://www.tutorialspoint.com/append-multiple-lists-at-once-in-python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


def remove_outliers(data):
    return data[(data['1st Flr SF'] < 4000) &
               ((data)['Gr Liv Area'] < 4000) &
               ((data)['Garage Yr Blt'] != 2207)]

# Thanks Will Badr for this! https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
def imp_data(data):
    has_nulls = data.isnull().mean() != 0
    null_columns = data.columns[has_nulls]
    for column in null_columns:
        try:
            data[column] + 1 # If this doesn't throw an error, it means it's an integer/float, and NaN values likely mean the value is 0
            data[column].fillna(0, inplace=True)
        except:
            data[column].fillna('NA', inplace=True)


def category_to_bool_cols(dataframe, list_of_columns):
    for column in list_of_columns:
        dummy_split = pd.get_dummies(dataframe[column], column, drop_first = True) # Creates dummy columns with the name {column}_{value_in_row} per get_dummies documentation
        for dummy_key in dummy_split: # Iterates through dummy_key in dummy_split
            dataframe[dummy_key] = dummy_split[dummy_key] # adds new columns named {dummy_key} to original dataframe

            
def log_cols(data, columns):
    change_0_to_1 = lambda x: 1 if x <= 0 else x
    for column in columns:
        temp_df = data[column].apply(change_0_to_1)
        data[f"log_{column.replace(' ', '_').lower()}"] = np.log(temp_df)

        
def log_hist(data, column):
    plt.hist(data.loc[:, column].apply(change_0_to_1))


def random_feature_thresh_test(data, target, features, threshold_start):
    best_threshold = 0
    best_score = float('inf')
    for i in range(0, 100):
        mean_corr = data.corr().loc[:, target].mean()
        feature_threshold = threshold_start + (i / 100)
        abs_value_greater_than_thresh = abs(data.corr().loc[:, 'SalePrice']) > mean_corr * feature_threshold
        # EdChum and dartdog from SO: https://stackoverflow.com/questions/29281815/pandas-select-dataframe-columns-using-boolean
        strong_corr_features = data.loc[:, data.corr().columns[abs_value_greater_than_thresh]]

        features = list(strong_corr_features[1:])
        features_not_in_list = ['SalePrice', 'PID', 'Id'
                               ]
        features = [feature for feature in features if feature not in features_not_in_list]

        X = data.loc[:,  features]
        y = data.loc[:,  target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=342)

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        lr.score(X_test, y_test)
        score = metrics.mean_squared_error(y_test, y_pred, squared=False)
        if score < best_score:
            print(score)
            print("^^NEW HIGH SCORE^^")
            best_score = score
            best_threshold = feature_threshold
    return f'The best score was {best_score}, the best threshold was {best_threshold}.'
    

def get_features(data, threshold):
    mean_corr = data.corr()['SalePrice'].mean()
    abs_value_greater_than_thresh = abs(data.corr()['SalePrice']) > mean_corr * threshold
    # EdChum and dartdog from SO: https://stackoverflow.com/questions/29281815/pandas-select-dataframe-columns-using-boolean
    strong_corr_features = data.loc[:,  data.corr().columns[abs_value_greater_than_thresh]]

    features = list(strong_corr_features[1:])
    features_not_in_list = ['SalePrice', 'PID', 'Id'
                           ]
    try:
        return [feature for feature in features if feature not in features_not_in_list]
    except:
        features_not_in_list = ['PID', 'Id'
                       ]
        return [feature for feature in features if feature not in features_not_in_list]


def get_cval_score_mse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=342)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    resids = y_test - y_pred
    print(f'The Cross Validation Score is: {cross_val_score(lr, X_train, y_train)}')
    print(f'The R2 score on testing data is: {lr.score(X_test, y_test)}')
    print(f'The MSE is {metrics.mean_squared_error(y_test, y_pred, squared=False)}')
    return X_train, X_test, y_train, y_test

def create_new_features(data):
    # Separating data into two groups, one pre 1982, one after 1982
    greater_than_1982 = data['Year Built'] > 1982
    data['built_1983_to_present'] = np.where(greater_than_1982, 1, 0)
    data['built_before_1983'] = np.where(~greater_than_1982, 1, 0)
    

# Creating my own polynomial features
def poly_features(data):
#     data['Avg_bsmt_kitch_exter_qual'] = (data['Bsmt Qual_TA'] + data['Kitchen Qual_TA'] + data['Exter Qual_TA']) * 2
    data['log_total_house_sf'] = data[['1st Flr SF', '2nd Flr SF', 'Total Bsmt SF', 'Wood Deck SF', 'Open Porch SF']].sum(axis=1)
    data['Overall Qual ^ 2'] = data['Overall Qual'] ** 2
#     data['total_house_sf_x_overall_qual'] = data[['1st Flr SF', '2nd Flr SF', 'Total Bsmt SF', 'Wood Deck SF', 'Open Porch SF']].sum(axis=1) * data['Overall Qual']
    # Thank you Jezrael from Stack Overflow! https://stackoverflow.com/questions/42063716/pandas-sum-up-multiple-columns-into-one-column-without-last-column

    
def remove_collinear_features(data):
    collinear_features = [
                          '1st Flr SF',
                          '2nd Flr SF',
                          'Total Bsmt SF',
                          'Wood Deck SF',
                          'Open Porch SF',
                          'Gr Liv Area',
                          'Lot Area', 
                         'BsmtFin SF 1',
                         'BsmtFin SF 2',
                         'Bsmt Unf SF',
                         'Low Qual Fin SF',
                          'Garage Yr Blt',
                          'Year Remod/Add',
                         ]
    columns_rm_collinear = [col for col in data.columns if col not in collinear_features]
    return columns_rm_collinear
    

def clean_train_data_export_csv(data, nominal_categories, categories_to_log):
    data = remove_outliers(data)
    create_new_features(data)
    imp_data(data)
    category_to_bool_cols(data, nominal_categories)
    poly_features(data)
    log_cols(data, categories_to_log)
    columns_rm_collinear = remove_collinear_features(data)
    return data[columns_rm_collinear]


def clean_test_data_export_csv(data, nominal_categories, categories_to_log):
    create_new_features(data)
    imp_data(data)
    category_to_bool_cols(data, nominal_categories)
    poly_features(data)
    log_cols(data, categories_to_log)
    columns_rm_collinear = remove_collinear_features(data)
    return data[columns_rm_collinear]