import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data, encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoding the dummy variable trap-- 3 dummy kolonun 2 tanesi algroitmada olmasi icin elimizle 2 tanesini sectik.
X = X[:, 1:] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # algoritmanın oluşturulması
regressor.fit(X_train, y_train) # algoritmanın eğitilme kısmı


# Predicting the Test set results
y_pred = regressor.predict(X_test)



#******** backward elemination *****************
import statsmodels.formula.api as sm  
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) #50 satir bir sutunluk 1 lerden olusan diziyi olusturuyor. X i yanına yapistiriyo, bunu tekrar x e atiyor.
#amac matematiksel denklemi koda dökmek

#p degeri 0.05 den buyuk olan kolonlari cikararak gidiyoruz.
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #her seyi gosteren bir fonksiyon bunun sayesinde p value yu kontrol ediyoruz.

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()