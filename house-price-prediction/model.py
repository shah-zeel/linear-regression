""" Import Libraries """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

""" Data """
# Read CSV file
HouseDF = pd.read_csv('USA_Housing.csv')

# create array
X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = HouseDF[['Price']]

# spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 

""" Creating and Training the LinearRegression Model """
# Saving object in a variable
lm = LinearRegression()
lm.fit(X_train, y_train)

""" LinearRegression Model Evaluation """
coeff_df = pd.DataFrame(lm.coef_.reshape(1, -1), columns=X.columns, index=['Coefficient'])

""" Predictions from our Linear Regression Model """
predictions = lm.predict(X_test)

# Graphs
# Create a folder to store the graphs
os.makedirs("graphs", exist_ok=True)

# Scatter plot
plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.title("Scatter Plot of Actual vs. Predicted Values")
plt.savefig("graphs/scatter_plot.png")
plt.close()

# Distribution plot
sns.distplot((y_test - predictions), bins=50)
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.title("Distribution of Residuals")
plt.savefig("graphs/distribution_plot.png")
plt.close()


""" Regression Evaluation Metrics """
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
