import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from dmba import textDecisionTree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # Create the parameter grid based on the results of random search 
%matplotlib inline 
# tells VS Code to display plots inside the interface

# Decision tree regression

# set X and y
X=spotify_df[X1cols]
y=spotify_df['charted_billboard']

print (X.shape)
print (y.shape)

from scipy.stats import skew

# pulls out only numeric data columns - in case there were any non-numeric ones
numeric_data = X.select_dtypes(include=[np.number])
numeric_data.boxplot(vert=False,figsize=(8,8))


# quantifies the skewedness of the numeric data columns
skewed = X[numeric_data.columns].apply(lambda x: skew(x.dropna().astype(float)))
skewed

numeric_data_small = numeric_data[numeric_data.columns[numeric_data.columns!='streams']]
numeric_data_small.boxplot(vert=False,figsize=(8,8))
plt.show()

# and extracts the index of only variables with a skew value of greater than .75
rskewed = skewed[skewed > 0.75 ].index # 'CRIM', 'ZN', 'DIS', 'RAD', 'LSTAT' features identified as skewed

# and extracts the index of only variables with a skew value of less than -.75
lskewed = skewed[skewed < -0.75 ].index # 'PTRATIO' feature identified as skewed

# and uses this index to map the right skewed variables
X[rskewed].hist(bins=20, figsize=(15, 7), color='lightblue', xlabelsize=0, ylabelsize=0, grid=False, layout=(2, 7))
plt.show()

# and uses this index to map the left skewed variables
X[lskewed].hist(bins=20, figsize=(15, 7), color='orange', xlabelsize=0, ylabelsize=0, grid=False, layout=(2, 7))
plt.show()

# log-transforms the highly right skewed variables of the dataset
X[rskewed] = np.log1p(X[rskewed])

# log-transforms the left skewed reflected variables of the dataset 
X[lskewed] = np.log1p(np.max(X[lskewed])-X[lskewed])

# now plot again the variables after their log transformation
X[rskewed].hist(bins=20,figsize=(15,7), color='lightblue', xlabelsize=0, ylabelsize=0, grid=False, layout=(2,7))
plt.show()

# now plot again the variables after their log transformation
X[lskewed].hist(bins=20,figsize=(15,7), color='orange', xlabelsize=0, ylabelsize=0, grid=False, layout=(2,7))
plt.show()

# scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

scaled_X.boxplot(vert=False,figsize=(8,8))

# split the dataset
(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled) = train_test_split(scaled_X, y, train_size=0.7)#, random_state=1)

from sklearn.tree import DecisionTreeRegressor
#define the model
regtree=DecisionTreeRegressor()
# fit model on training data
regtree.fit(X_train_scaled,y_train_scaled)

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
# report accuracy score for linear regression model
y_pred_scaled=regtree.predict(X_test_scaled)
print ('MSE: ',mean_squared_error(y_test_scaled,y_pred_scaled))
print ('R2: ', r2_score(y_test_scaled,y_pred_scaled))

# non-scaled (raw data)

# split the dataset
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7)#, random_state=1)

from sklearn.tree import DecisionTreeRegressor
#define the model
regtree=DecisionTreeRegressor()
# fit model on training data
regtree.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
# report accuracy score for linear regression model
y_pred=regtree.predict(X_test)
print ('MSE: ',mean_squared_error(y_test,y_pred))
print ('R2: ', r2_score(y_test,y_pred))

# let's see what features are the most important for prediction of the dependent variable MEDV
plot_feature_importance(regtree.feature_importances_, X_train_scaled.columns, 'REGRESSION DECISION TREE')
plot_feature_importance(regtree.feature_importances_, X_train.columns, 'REGRESSION DECISION TREE')