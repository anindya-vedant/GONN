import pandas as pd

data=pd.read_csv('wbc-dataset.csv',header=None)
data=data.drop(data.columns[0],axis=1)
x=data.iloc[:,:-1].values
y=data.iloc[:,9].valuesord
print(x)
print(y)