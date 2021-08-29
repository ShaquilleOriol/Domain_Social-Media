import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plot
from matplotlib.pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data Collection and reading
OnlineNewsDataframe = pd.read_csv("OnlineNewsPopularity.csv")
OnlineNewsDataframe.head(6)


# Analysing data & finding dependent and independent variables
X = OnlineNewsDataframe.iloc[:, 1:60]
X.head()

corr = X.corr()
sns.heatmap(corr)
rcParams['figure.figsize'] = 20, 20

Y = OnlineNewsDataframe["shares"]
Y.head()


# Splitting data into training and testing
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, random_state=5, test_size=0.20)


# Initializing model and training model
linearModel = LinearRegression()
linearModel.fit(train_x, train_y)

# Predict data
prediction = linearModel.predict(test_x)

# Mean Squared errors
metrics.mean_squared_error(prediction, test_y)

# Plotting predicted and test data
plot.scatter(prediction, test_y)
plot.show()

