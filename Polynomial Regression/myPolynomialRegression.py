import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #burda 1. sutunu almak istiyoruz. direk 1 yazabilirdik ama boyle yazinca matris halinde tutuyor x i
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  #karsilastirma yapmak icin linear regresyon yapiyoruz


#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3) #polinomun derecesiyle alakali olarak degree degeri veriyoruz
X_poly = poly_reg.fit_transform(X) #polinomun ozelliklerini cikarip sutunlari olusturuyor.

#burda tekrar linear regresyon yaparak polinomal regresyonu ssagliyor.
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#visualling the linear regression results (bu model duzgun sonuclar vermeyecektir)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('truth or bluff(linear regression)')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()

#visualling the polynomial regression results 
X_grid = np.arange(min(X), max(X), 0.1) #grafikte mavi tahmin cizgilernin  x eksenine göre 0.1 mesafede bir tahmin yapilarak daha dogru ve goze hitap eden sonuclar vermesini saglar
X_grid = X_grid.reshape((len(X_grid), 1)) # x_grid i , asagidaki plt.plot ta x yerlerine yazdigin an daha duzgun graph ortaya cikar

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') #predictin icine uzun uzun yazma sebebimiz, yeni bir matris icin de direk matris adini degistirip ayni islemi yapabilmek
#istesek predictin icine X_poly yazabilirdik ama o zaman sadece var olan X matrisi icin yazmis olacaktir genel bir kod olmaycakti.
plt.title('truth or bluff(polynomial regression)')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()

#predixting a new result with linear regression 
lin_reg.predict(6.5)   #6.5 level tecrubeli birinin maasini linear regresyona gore tahmin ettik, sonuc kotu

#predicting a new result with polynomial regression 
lin_reg_2.predict(poly_reg.fit_transform(6.5)) #6.5 levellik birinin maasini polinomal regresyona gore tahmin ettik, sonuc iyi
