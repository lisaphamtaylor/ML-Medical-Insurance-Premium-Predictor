from data import df
from sklearn import linear_model, model_selection, metrics, svm, ensemble
from xgboost import XGBRegressor
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# df.info()
X=df.drop('expenses',axis=1)
Y=df[['expenses']]

# print(X,Y)

list_1 = []
list_2 = []
list_3 = []

cross_value_score = 0

'''training and validating model'''
for i in range(0, 100):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=i)
    lrmodel = linear_model.LinearRegression()
    lrmodel.fit(X_train, Y_train)
    
    list_1.append(lrmodel.score(X_train, Y_train))
    list_2.append(lrmodel.score(X_test, Y_test))

    cross_value_score = (model_selection.cross_val_score(lrmodel, X, Y, cv=5)).mean()
    list_3.append(cross_value_score)

    df1 = pd.DataFrame({'Train Score': list_1, 'Test Score': list_2, 'Cross Value Score': list_3})

'''after identifying 42 is the best random_state number for this data set, we fix it to try different ML algos for better accuracy/scoring'''
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,Y,test_size=0.2,random_state=42)

'''printing out the r2 and cvs scores'''
def r2_cvs_scores(model, y_pred_train, y_pred_test):
    print(f'Training model score: {metrics.r2_score(ytrain,y_pred_train): >20}')
    print(f'Test model score: {metrics.r2_score(ytest,y_pred_test): >24}')
    print(f'CVS: {model_selection.cross_val_score(model,X,Y,cv=5,).mean(): >37} \n')

'''Linear Regression Model'''
def Linear_Regression_Model(X,Y):
    lrmodel = linear_model.LinearRegression()
    lrmodel.fit(xtrain,ytrain)
    print('\nLinear Regression Model')
    print(f'Training model score: {lrmodel.score(xtrain,ytrain): >20}')
    print(f'Test model score: {lrmodel.score(xtest,ytest): >24}')
    print(f'CVS: {model_selection.cross_val_score(lrmodel,X,Y,cv=5,).mean(): >37} \n')
# Linear_Regression_Model(X,Y)

'''SVR Model'''
def SVR_Model(X,Y):
    svr_model = svm.SVR()
    svr_model.fit(xtrain,ytrain)
    y_pred_train1 = svr_model.predict(xtrain)
    y_pred_test1 = svr_model.predict(xtest)
    print('SVR Model')
    r2_cvs_scores(svr_model, y_pred_train1, y_pred_test1)
# SVR_Model(X,Y)

'''Random Forest Regressor Model'''
def Random_Forest_Regressor_Model(X,Y):
    rf_model = ensemble.RandomForestRegressor(random_state = 42)
    rf_model.fit(xtrain,ytrain)
    y_pred_train2 = rf_model.predict(xtrain)
    y_pred_test2 = rf_model.predict(xtest)
    print('Random Forest Regressor Model')
    r2_cvs_scores(rf_model, y_pred_train2, y_pred_test2)

    estimator = ensemble.RandomForestRegressor(random_state=42)
    param_grid = {'n_estimators':[10,40,50,98,100,120,150]}
    grid = model_selection.GridSearchCV(estimator, param_grid, scoring='r2', cv=5)
    grid.fit(xtrain, ytrain)
    print(grid.best_params_, f'{grid.best_score_=}')
    best_params = grid.best_params_
    rf_model = ensemble.RandomForestRegressor(random_state=42, n_estimators=best_params['n_estimators'])
    rf_model.fit(xtrain,ytrain)
    y_pred_train2 = rf_model.predict(xtrain)
    y_pred_test2 = rf_model.predict(xtest)
    print('--hyperparameter tuning--')
    r2_cvs_scores(rf_model, y_pred_train2, y_pred_test2)

# Random_Forest_Regressor_Model(X,Y)

'''Gradient Boosting Regressor Model'''
def Gradient_Boosting_Regressor_Model(X,Y):
    gb_model = ensemble.GradientBoostingRegressor()
    gb_model.fit(xtrain,ytrain)
    y_pred_train3 = gb_model.predict(xtrain)
    y_pred_test3 = gb_model.predict(xtest)
    print('Gradient Boosting Regressor Model')
    r2_cvs_scores(gb_model, y_pred_train3, y_pred_test3)

    estimator = ensemble.GradientBoostingRegressor()
    param_grid = {'n_estimators':[10,15,19,20,21,50],'learning_rate':[0.1,0.19,0.2,0.21,0.8,1]}
    grid = model_selection.GridSearchCV(estimator, param_grid, scoring='r2', cv=5)
    grid.fit(xtrain, ytrain)
    print(grid.best_params_, f'{grid.best_score_=}')
    best_params = grid.best_params_
    gb_model = ensemble.GradientBoostingRegressor(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'])
    gb_model.fit(xtrain,ytrain)
    y_pred_train3 = gb_model.predict(xtrain)
    y_pred_test3 = gb_model.predict(xtest)
    print('--hyperparameter tuning--')
    r2_cvs_scores(gb_model, y_pred_train3, y_pred_test3)

# Gradient_Boosting_Regressor_Model(X,Y)

'''XGBoost Regressor Model'''
def XGBoost_Regressor_Model(X,Y):
    xg_model = XGBRegressor()
    xg_model.fit(xtrain,ytrain)
    y_pred_train4 = xg_model.predict(xtrain)
    y_pred_test4 = xg_model.predict(xtest)
    print('XGBoost Regressor Model')
    r2_cvs_scores(xg_model, y_pred_train4, y_pred_test4)

    estimator = XGBRegressor()
    param_grid = {'n_estimators':[10,15,20,40,50],'max_depth':[3,4,5],'gamma':[0,0.15,0.3,0.5,1]}
    grid = model_selection.GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
    grid.fit(xtrain,ytrain)
    print(grid.best_params_, f'{grid.best_score_=}')
    best_params = grid.best_params_
    xg_model = XGBRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], gamma=best_params['gamma'])
    xg_model.fit(xtrain,ytrain)
    y_pred_train4 = xg_model.predict(xtrain)
    y_pred_test4 = xg_model.predict(xtest)
    print('--hyperparameter tuning--')
    r2_cvs_scores(xg_model, y_pred_train4, y_pred_test4)

# XGBoost_Regressor_Model(X,Y)
