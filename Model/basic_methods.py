import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb  
import statistics
from scipy.stats import norm 

# method to get counts for each price range 
def get_prices_count(prices):
    raw_price = {} 
    count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = 0
    for price in prices: 
        if price > 100 and price < 150: 
            count1 += 1
        elif price > 150 and price < 200: 
            count2 += 1
        elif price > 200 and price < 250: 
            count3 += 1 
        elif price > 250 and price < 300: 
            count4 += 1
        elif price > 300 and price < 350: 
            count5 += 1
        elif price > 350 and price < 400: 
            count6 += 1
        elif price > 400 and price < 450: 
            count7 += 1
        elif price > 450 and price < 500:
            count8 += 1
        else: 
            count9 += 1

    raw_price['1-1.5'] = count1 
    raw_price['1.5-2'] = count2 
    raw_price['2-2.5'] = count3 
    raw_price['2.5-3'] = count4 
    raw_price['3-3.5'] = count5 
    raw_price['3.5-4'] = count6 
    raw_price['4-4.5'] = count7 
    raw_price['4.5-5'] = count8 
    raw_price['5+'] = count9

    return raw_price 

# plot counts of raw price information 
def plot_raw_prices(price_dict): 
    price_ranges = price_dict.keys() 
    counts = price_dict.values() 
    plt.bar(price_ranges, counts)
    plt.title('Raw Price Ranges')
    plt.xlabel('Price Range (Hundreds)') 
    plt.ylabel('Count') 
    plt.show()

# method to plot Airbnb locations by latitude and longitude 
def plot_points_by_latitude(lats, longs, data):
    plt.figure(figsize=(12,12))
    sb.jointplot(x=lats, y=longs, size=12) 
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show() 

# method to plot by location and price 
def plot_points_by_location_and_price(data):
    lats = list(map(float, data['latitude']))
    longs = list(map(float, data['longitude']))
    prices = list(map(int, data['price']))
    colors = [] 
    for i in range(len(prices)):
        if prices[i] < 100:
            colors.append('green')
        elif prices[i] > 100 and prices[i] < 300: 
            colors.append('yellow') 
        elif prices[i] > 300 and prices[i] < 500: 
            colors.append('orange') 
        else: 
            colors.append('red')

    sb.jointplot(x=lats, y=longs, 
        joint_kws= {
            "color": colors, 
        }) 
    plt.xlabel('Latitude')
    plt.ylabel('Longitude') 
    plt.show() 

# goal to find average price of each region and plot these points on a scatterplot 
def calculate_prices_by_region(region_set, prices, regions):
    price_table = {} 
    for region in region_set: 
        price_table[region] = []
        #print('hello')

    for i in range(0,len(prices)):
        price_table[regions[i]].append(prices[i])

    for region in price_table.keys():
        avg = statistics.mean(price_table[region])
        if len(price_table[region]) == 1: 
            sd = 0
        else: 
            sd = statistics.stdev(price_table[region])
        price_table[region] = (avg, sd) 

    return price_table

# plot bar graph of different housing types 
def plot_housing_types(data):
    counts = [] 
    room_types = set(data['room_type'])
    count1 = count2 = count3 = count4 = 0 
    for type in data['room_type']:
        if str(type) == 'Entire home/apt':
            count1 += 1
        elif str(type) == 'Hotel room': 
            count2 += 1
        elif str(type) == 'Shared room': 
            count3 += 1
        elif str(type) == 'Private room':
            count4 += 1

    counts.append(count1) 
    counts.append(count2) 
    counts.append(count3) 
    counts.append(count4) 

    plt.pie(counts, labels=['Entire home/apt', 'Hotel/Shared room', '', 'Private room'])
    plt.title('Housing Counts') 
    plt.show() 

# plot overall satisfaction on bar graph
def plot_ratings(data):
    counts = [] 
    ratings = data['review_scores_rating']
    count1 = count2 = count3 = count4 = count5 = 0 
    for rating in ratings: 
        if rating > 4.5:
            count1 += 1
        elif rating > 4.0 and rating < 4.5: 
            count2 += 1
        elif rating > 3.5 and rating < 4.0: 
            count3 += 1
        elif rating > 3.0 and rating < 3.5: 
            count4 += 1
        elif rating < 3.0:
            count5 += 1
    counts.append(count1) 
    counts.append(count2) 
    counts.append(count3) 
    counts.append(count4) 
    counts.append(count5) 

    rating_list = ['>4.5', '4-4.5', '3.5-4', '3-3.5', '<3']
    plt.bar(rating_list, counts) 
    plt.title('Rating Counts') 
    plt.xlabel('Ranges') 
    plt.ylabel('Counts') 
    plt.show()    

# plot counts of accommodates on bar graph 
def plot_accommodates(data):
    counts = [] 
    accommodates = data['accommodates']
    accommodates_set = set(accommodates)
    count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = count10 = count11 = count12 = count13 = count14 = count15 = count16 = 0
    for num in accommodates:
        if num == 1: 
            count1 += 1
        elif num == 2: 
            count2 += 1
        elif num == 3: 
            count3 += 1 
        elif num == 4: 
            count4 += 1
        elif num == 5: 
            count5 += 1
        elif num == 6: 
            count6 += 1
        elif num == 7: 
            count7 += 1
        elif num == 8: 
            count8 += 1 
        elif num == 9: 
            count9 += 1
        elif num == 10: 
            count10 += 1
        elif num == 11: 
            count11 += 1
        elif num == 12: 
            count12 += 1
        elif num == 13: 
            count13 += 1
        elif num == 14: 
            count14 += 1
        elif num == 15:
            count15 += 1 
        elif num == 16:
            count16 += 1
        else: 
            count17 += 1

    plt.bar(list(accommodates_set), [count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12, count13, count14, count15, count16])
    plt.title('Accommodates Counts') 
    plt.xlabel('Accommodates') 
    plt.ylabel('Counts') 
    plt.show() 

# plot normal distribution of price 
def plot_price_avg_and_sd(prices):   
    avg_price = statistics.mean(prices)
    sd_price = statistics.stdev(prices) 
    x_axis = np.arange(-5000, 6000, 1)
    plt.plot(x_axis, norm.pdf(x_axis,avg_price,sd_price))
    plt.title('Normal Distribution of Listing Prices') 
    plt.xlabel('Prices')
    plt.ylabel('Probability')
    plt.show() 

# remove outliers 
def remove_outliers(data, sds):
    for column in data:  
        mean = data[column].mean() 
        sd = data[column].std() 

        data = data[(data[column] <= mean + (sds*sd))]

    return data 

# testing
region_set1 = {'Alabama', 'California'}
prices = [69, 35, 420, 37]
regions = ['Alabama', 'Alabama', 'California', 'Alabama']
print(calculate_prices_by_region(region_set1, prices, regions))




