import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""getting database info and checking values"""
df=pd.read_csv('insurance.csv')
# print(df)              # prints the whole database
# df.info()              # gives info about the database table
# print(df.describe())   # gives statistical info

"""Exploratory Data Analysis [EDA]"""
'''checking for trends and null values'''
# print(df.isnull().sum())        # checks for null values

'''plotting out data'''
def plot_pie_chart(dataset, features=None):
    if features is None:
        features = dataset.columns
    
    for i, column in enumerate(features):
        plt.subplot(1,len(features),i+1)

        x = dataset[column].value_counts()  # counts the number of values in the column
        plt.pie(x.values,           # sets values of plot
                labels=x.index,     # shows labels
                autopct='%.2f%%',   # shows percentage
                labeldistance=0.3)
    plt.show()

def plot_bar_chart(dataset, features=None):
    if features is None:
        features = dataset.columns

    for i, column in enumerate(features):
        sub_df = dataset.groupby([column])['expenses'].mean()
        plt.subplot(len(features)//2, 2, i+1)
        ax = sub_df.plot.bar(x=column, rot=90)
    
    plt.show()

def plot_scatter_chart(dataset, features=None):
    if features is None:
        features = dataset.columns
    
    for i, column in enumerate(features):
        plt.subplot(len(features)//2, 2, i+1)
        # dataset.plot.scatter(x=column, y='expenses', )
        sns.scatterplot(data=dataset, x=column, y='expenses', hue='smoker')
    plt.show()

# plot_pie_chart(df,['sex','smoker', 'region'])
# plot_bar_chart(df, ['sex','children','smoker','region'])
# plot_scatter_chart(df, ['age','bmi'])

"""DATA PREPROCESSING"""
df.drop_duplicates(inplace=True)    # drops duplicate values
# sns.boxplot(df['bmi'])       # shows outliers
# plt.show()
'''caclulating IQR to determine outlier caps'''
Q1=df['bmi'].quantile(0.25)
Q2=df['bmi'].quantile(0.5)
Q3=df['bmi'].quantile(0.75)
iqr=Q3-Q1
lowlim=Q1-1.5*iqr   # lower limit as set by IQR
upplim=Q3+1.5*iqr   # upper limit as set by IQR

df['bmi'] = df['bmi'].clip(lower=lowlim, upper=upplim)        # caps outliers to normalize values for model training
# sns.boxplot(df['bmi'])       # shows outliers
# plt.show()

'''encoding data--converting categorical data to numerical data'''
df['sex']=df['sex'].map({'male':0,'female':1})
df['smoker']=df['smoker'].map({'yes':1,'no':0})
df['region']=df['region'].map({'northwest':0, 'northeast':1,'southeast':2,'southwest':3})

# print(df.corr())        # prints correlation mx

