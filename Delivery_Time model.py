
# Step -1 Import data files
import numpy as np
import pandas as pd

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Simple Linear Regression\\delivery_time.csv")
df
df.head()
df.shape
df.isnull().sum()
df.describe()

# Step - 2 Split the variables in X and Y
Y = df[["Delivery Time"]]
X = df[["Sorting Time"]]

#EDA
 # Scatter Plot
import matplotlib.pyplot as plt
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

# Boxplot
plt.boxplot(df["Delivery Time"])
plt.boxplot(df["Sorting Time"])

# Histogram
plt.hist(df["Delivery Time"],bins=5)
plt.hist(df["Sorting Time"],bins=5)

df.corr()
# Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_ #Bo
LR.coef_ #B1

# Predict the value
Y_pred = LR.predict(X)
Y_pred

# Scatter Plot with Plot
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],Y_pred,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(Y, Y_pred).round(3)*100)

"""
There is a Difference of RMSE is 2.792 and the R2 is 0.682
"""

# Transformation
# Model 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1 = LR.predict(np.log(X))

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

"""
There is a Difference of RMSE is 2.733 and the R2 is 0.695
"""

# Model 3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1 = LR.predict(np.sqrt(X))

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

"""
There is a Difference of RMSE is 2.732 and the R2 is 0.696
"""

# Model 4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1 = LR.predict(X**2)

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

"""
There is a Difference of RMSE is 3.011 and the R2 is 0.630
"""

# Model 5

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**3,Y)
y1 = LR.predict(X**3)

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

"""
There is a Difference of RMSE is 3.253 and the R2 is 0.5689
"""

"""
Inference : Delivery time is predicted using sorting time and the best model selected is model 3
which is transformed using squared tranformation because the graph is bell shaped and its rscore is 0.69580.
"""













    

