import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb 
from scipy.spatial import distance 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats 
from basic_methods import * 

# read in select columns 
excel_file = 'f2022_datachallenge.csv'
use_these = ['host_location', 'host_neighbourhood', 'latitude', 'longitude', 'room_type', 'accommodates', 'bedrooms', 'beds', 'price', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy']
data = pd.read_csv(excel_file, usecols=use_these)

data['room_type'] = data['room_type'].replace(['Shared room', 'Private room', 'Hotel room', 'Entire home/apt'], [1, 2, 3, 4])

neighborhoods = set(data['host_neighbourhood'])
num_neighborhoods = len(neighborhoods)
data['host_neighbourhood'] = data['host_neighbourhood'].replace(list(neighborhoods), list(range(num_neighborhoods)))

regions = data['host_location']
bad_labels = [] 
for index, region in enumerate(regions): 
    if "Hawaii" not in str(region) and "HI" not in str(region): 
        bad_labels.append(index)
data = data.drop(labels=bad_labels, axis=0)

data.drop('host_location', inplace=True, axis=1) 

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

data.replace('', 0, inplace=True) 
data.dropna(inplace=True) 

data = remove_outliers(data, 3)

corrs = data.corr() 
price_corrs = corrs['price']
print(price_corrs) 

corrmat = data.corr(method = 'spearman') 
f, ax = plt.subplots(figsize=(12,9))
sb.heatmap(corrmat, vmax=0.8, square=True)

temp = pd.DataFrame(preprocessing.scale(data, with_mean=False))
test_metrics = data 
target = pd.DataFrame(data, columns=["price"])
temp.columns = test_metrics.columns 
test_metrics = temp 

metrics_train, metrics_test, target_train, target_test = train_test_split(test_metrics, target, test_size=0.2)
 
knn = KNeighborsRegressor(n_neighbors=6, algorithm='brute') # set to brute force algorithm 
cols = ['accommodates', 'beds', 'bedrooms', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy'] # specify parameters
knn.fit(metrics_train[cols], target_train['price'])     # fitting curve over trained set of metrics and target 

predictions = knn.predict(metrics_test[cols]) 
predictions_mse = mean_squared_error(target_test['price'], predictions) 
predictions_rmse = predictions_mse**(1/2)
print(predictions_rmse)

actual_prices = data['price'][8314:]
plt.scatter(predictions, actual_prices)
plt.title('Predicted Price vs. Actual Price') 
plt.xlabel('Predicted Price in dollars') 
plt.ylabel('Actual Price in dollars') 

z = np.polyfit(predictions, actual_prices, 1)
p = np.poly1d(z)
plt.plot(predictions, p(predictions), color="red", linewidth=3, linestyle="--")

plt.show() 


