# Importing all necessary libraries
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

# Importing IRIS dataset from sklearn
iris = datasets.load_iris()
# print(iris)

# Splitting data into Input and output objects
X = iris.data  # Input
y = iris.target  # Output

# Splitting data into Training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Initiating the Linear_regression model object
linear_regression_model = LinearRegression()

# Initiating the Logistic_regression model object
logistic_regression_model = LogisticRegression()

# Initiating the Support Vector Classifier model object
svc_model = SVC()

# Linear regression Model training
lin_reg = linear_regression_model.fit(x_train, y_train)

# Logistic regression Model training
log_reg = logistic_regression_model.fit(x_train, y_train)

# Support Vector Classifier Model training
svc_model = svc_model.fit(x_train, y_train)

# We will dump/save ML models into the respective pickle files. Here 'wb' shows the file is in the write mode
pickle.dump(lin_reg, open('linear_regression_model.pkl', 'wb'))
pickle.dump(log_reg, open('logistic_regression_model.pkl', 'wb'))
pickle.dump(svc_model, open('svm_model.pkl', 'wb'))
