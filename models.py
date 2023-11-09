from data import df
from sklearn import linear_model, model_selection
import pandas as pd


df.info()
X=df.drop('expenses',axis=1)
Y=df[['expenses']]

print(X,Y)

list_1 = []
list_2 = []
list_3 = []

cross_value_score = 0

for i in range(0, 100):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=i)
    lrmodel = linear_model.LinearRegression()
    lrmodel.fit(X_train, Y_train)
    
    list_1.append(lrmodel.score(X_train, Y_train))
    list_2.append(lrmodel.score(X_test, Y_test))

    cross_value_score = (model_selection.cross_val_score(lrmodel, X, Y, cv=5)).mean()
    list_3.append(cross_value_score)

    df1 = pd.DataFrame({'Train Score': list_1, 'Test Score': list_2, 'Cross Value Score': list_3})

    print(df1)