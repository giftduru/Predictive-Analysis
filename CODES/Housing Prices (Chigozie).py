
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:14:33 2020

@author: Prince Igweze and Gift Duru
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
from numpy import std

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

#EXPLORATORY DATA ANALYSIS

#Study Data
data_train.head()

#Study Statistics of Sales Price
print(data_train['SalePrice'].describe())


#Correlation Matrix
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, linewidth=1,linecolor='k', square=True);

#Extract Features with High Correlation with Sales Price
#Understand how data is distributed overall
feature_list=corrmat['SalePrice'][:-1][abs(corrmat['SalePrice'][:-1])>0.5].sort_values(ascending=False)
print(feature_list)

#Examining categorical variables
#categorical_features = data_train.select_dtypes(include=[np.object])
#Column-wise distribution of null values
#categorical_features = categorical_features.fillna(value = 0)
#print(categorical_features.isnull().sum())
#cat = categorical_features.apply(lambda x: pd.factorize(x)[0])
## ANOVA testing between categorical variables and 

#from scipy import stats
#F, p = stats.f_oneway(data_train[data_train=='MSZoning'].SalePrice)

#from scipy import stats
#F, p = stats.f_oneway(data_train[data_train=='MSZoning'].SalePrice, data_train[data_train=='Alley'].SalePrice)
#print(F)
#Cat_var = pd.factorize(categorical_features)
#print(Cat_var)
#print(len(categorical_features.columns))
#print(len(data_train.columns))


# Estimate of Location and Variability (Central Tendency)
print(data_train['OverallQual'].describe())
print(data_train['GrLivArea'].describe())
print(data_train['GarageCars'].describe())
print(data_train['GarageArea'].describe())
print(data_train['TotalBsmtSF'].describe())
print(data_train['1stFlrSF'].describe())
print(data_train['FullBath'].describe())
print(data_train['TotRmsAbvGrd'].describe())
print(data_train['YearBuilt'].describe())
print(data_train['YearRemodAdd'].describe())

#DATA CLEANING 
#Missing Values
print(feature_list.isnull().sum())
#No missing value in features

#Identifying and Removing Outliers
#Visualizing Outliers Using Scattered Plots

# =============================================================================
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['OverallQual'], y=data_train['SalePrice'])
plt.title('OverallQual')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['GrLivArea'], y=data_train['SalePrice'])
plt.title('GrLivArea')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['GarageCars'], y=data_train['SalePrice'])
plt.title('GarageCars')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['TotalBsmtSF'], y=data_train['SalePrice'])
plt.title('TotalBsmtSF')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['1stFlrSF'], y=data_train['SalePrice'])
plt.title('1stFlrSF')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['FullBath'], y=data_train['SalePrice'])
plt.title('FullBath')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['YearBuilt'], y=data_train['SalePrice'])
plt.title('YearBuilt')
 
fig = plt.figure(figsize=(15,15))
plt.scatter(x=data_train['YearRemodAdd'], y=data_train['SalePrice'])
plt.title('YearRemodAdd')
# =============================================================================


#Removing Outliers Using Standard Devation Method
#Calculate summary statistics (1)
data_mean, data_std = mean(data_train['OverallQual']), std(data_train['OverallQual'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['OverallQual'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['OverallQual'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (2)
data_mean, data_std = mean(data_train['GrLivArea']), std(data_train['GrLivArea'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['GrLivArea'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['GrLivArea'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))



#Calculate summary statistics (3)
data_mean, data_std = mean(data_train['GarageCars']), std(data_train['GarageCars'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['GarageCars'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['GarageCars'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (4)
data_mean, data_std = mean(data_train['GarageArea']), std(data_train['GarageArea'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['GarageArea'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['GarageArea'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (4)
data_mean, data_std = mean(data_train['TotalBsmtSF']), std(data_train['TotalBsmtSF'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['TotalBsmtSF'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['TotalBsmtSF'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (5)
data_mean, data_std = mean(data_train['1stFlrSF']), std(data_train['1stFlrSF'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['1stFlrSF'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['1stFlrSF'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (6)
data_mean, data_std = mean(data_train['FullBath']), std(data_train['FullBath'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['FullBath'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['FullBath'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (7)
data_mean, data_std = mean(data_train['TotRmsAbvGrd']), std(data_train['TotRmsAbvGrd'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['TotRmsAbvGrd'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['TotRmsAbvGrd'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (8)
data_mean, data_std = mean(data_train['YearBuilt']), std(data_train['YearBuilt'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['YearBuilt'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['YearBuilt'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (9)
data_mean, data_std = mean(data_train['YearRemodAdd']), std(data_train['YearRemodAdd'])

#Identify Outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data_train['YearRemodAdd'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_train['YearRemodAdd'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))



# TEST DATASET CLEANUP
test_features = data_test[["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]
#print(test_features.isnull().sum())
#if test_features.isnull():
test_features.fillna(0,inplace = True)
print(test_features.isnull().sum())

#Removing Outliers Using Standard Devation Method
#Calculate summary statistics (1)
dtest_mean, dtest_std = mean(data_test['OverallQual']), std(data_test['OverallQual'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['OverallQual'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['OverallQual'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (2)
dtest_mean, dtest_std = mean(data_test['GrLivArea']), std(data_test['GrLivArea'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['GrLivArea'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['GrLivArea'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))


#Calculate summary statistics (3)
dtest_mean, dtest_std = mean(data_test['GarageCars']), std(data_test['GarageCars'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['GarageCars'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['GarageCars'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (4)
dtest_mean, dtest_std = mean(data_test['GarageArea']), std(data_test['GarageArea'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['GarageArea'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['GarageArea'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (5)
dtest_mean, dtest_std = mean(data_test['TotalBsmtSF']), std(data_test['TotalBsmtSF'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['TotalBsmtSF'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['TotalBsmtSF'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (6)
dtest_mean, dtest_std = mean(data_test['1stFlrSF']), std(data_test['1stFlrSF'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['1stFlrSF'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['1stFlrSF'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (7)
dtest_mean, dtest_std = mean(data_test['FullBath']), std(data_test['FullBath'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['FullBath'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['FullBath'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (8)
dtest_mean, dtest_std = mean(data_test['TotRmsAbvGrd']), std(data_test['TotRmsAbvGrd'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['TotRmsAbvGrd'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['TotRmsAbvGrd'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (9)
dtest_mean, dtest_std = mean(data_test['YearBuilt']), std(data_test['YearBuilt'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['YearBuilt'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['YearBuilt'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#Calculate summary statistics (10)
dtest_mean, dtest_std = mean(data_test['YearRemodAdd']), std(data_test['YearRemodAdd'])

#Identify Outliers
test_cutoff = dtest_std * 3
lower, upper = dtest_mean - test_cutoff, dtest_mean + test_cutoff
outliers = [x for x in data_test['YearRemodAdd'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

#Remove Outliers
outliers_removed = [x for x in data_test['YearRemodAdd'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#TRAINING MODEL USING RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

#Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state = 42)

#Train the model on training data 
y = data_train["SalePrice"]
x = data_train[["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]
train_model = rf.fit(x,y);


#Making Prediction on the Test Set
train_features = data_train[["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]
train_predictions = rf.predict(train_features)
print(y.values)
print(train_predictions)

#Calculating the RSME
import math

RMSE = math.sqrt(np.mean((y.values - train_predictions)**2))
print('Root Mean Square Error: %d' % (RMSE))

#Predicting New Dataset
test_predictions = rf.predict(test_features)
print(test_predictions)

#f = np.concatenate((test_predictions, data_test['Id']))
#np.savetxt('Housing_Prices_Result.csv', X = (test_predictions), header = 'SalePrice')


#===============================================
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(test_features.data,test_features.target,train_size = .7)
# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# =============================================================================
