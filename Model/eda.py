import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb 
from scipy.spatial import distance 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from basic_methods import *  

# read in from the csv file 
excel_file = 'f2022_datachallenge.csv'
data = pd.read_csv(excel_file)

# remove all listings not in Hawaii
regions = data['host_location']
bad_labels = [] 
for index, region in enumerate(regions): 
    if "Hawaii" not in str(region) and "HI" not in str(region): 
        bad_labels.append(index)

data = data.drop(labels=bad_labels, axis=0)

accommodates = set(data['accommodates'])
prices = data['price']
regions = list(data['host_location'])
prices = list(data['price'])
region_set = set(regions) 

raw_price = get_prices_count(prices) 

# execute basic methods
plot_price_avg_and_sd(prices) 
plot_raw_prices(raw_price)
print(calculate_prices_by_region(region_set, prices, regions))
plot_points_by_latitude(data['latitude'], data['longitude'], data) 
plot_housing_types(data) 
plot_ratings(data) 
plot_accommodates(data)
plot_points_by_location_and_price(data) 
