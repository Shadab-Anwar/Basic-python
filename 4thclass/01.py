import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
df = pd.read_csv('./SALES.csv') #Importing the data set
new_df = df[['TV','SALES']]
X = np.array(new_df[['TV']]) # Storing into X the 'Engine HP' as np.array
y = np.array(new_df[['SALES']]) # Storing into y the 'MSRP' as np.array
print(X.shape) # Vewing the shape of X
print(y.shape) # Vewing the shape of y
plt.scatter(X,y,color="red") # Plot a graph X vs y
plt.title('Population of the city vs. Profit of restaurants')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=15)
regressor = LinearRegression() # Creating a regressior
print("Training Set:")
print(X_train)
print("Test Set :")
print(X_test)
regressor.fit(X_train,y_train) # Fiting the dataset into the model
plt.scatter(X_test,y_test,color="green") # Plot a graph with X_test vs y_test
plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3) # Regressior line showing
plt.title('Regression(Test Set)')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
y_pred = regressor.predict(X_test)
print('R2 score: %.2f' % r2_score(y_test,y_pred)) # Priniting R 2 Score
print('Mean Error :',mean_squared_error(y_test,y_pred)) # Priniting the mean error
