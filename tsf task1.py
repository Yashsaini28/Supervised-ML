import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x=pd.read_csv('student_scores - student_scores.csv')
x.shape
x.head()


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
x.describe()

x.plot(x='Hours', y='Scores', style='*')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

line = lr.coef_*X+lr.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.plot(X, line);

plt.show()

X =x.iloc[:, :1].values
y = .iloc[:, 1].values
#y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
X_test


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

hours = 9.25
my_prediction =lr.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(my_prediction[0]))

hours = 9.5
my_prediction =lr.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(my_prediction[0]))

hours = 5.4
my_prediction =lr.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(my_prediction[0]))

#Evaluation

lr.score(X_test,y_test)

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

      