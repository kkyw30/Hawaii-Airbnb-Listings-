from random import random
import spacy 
import pandas as pd 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
import statistics 

# read in select columns 
excel_file = 'f2022_datachallenge.csv'
use_these = ['name', 'description', 'neighborhood_overview', 'host_location', 'host_about', 'host_neighbourhood', 'review_scores_rating']
data = pd.read_csv(excel_file, usecols=use_these)

# remove all listings not in Hawaii
regions = data['host_location']
bad_labels = [] 
for index, region in enumerate(regions): 
    if "Hawaii" not in str(region) and "HI" not in str(region): 
        bad_labels.append(index)

data = data.drop(labels=bad_labels, axis=0)

data.isna().sum() 
data.dropna(inplace=True)

# text summarizer function 
def text_summarizer(text):
    apos_dict = {"'s": " is", "n't":" not", "'m":" am", "'ll":" will", "'d":" would", "'ve":" have", "'re":" are"}
    useless = ['the', 'a', 'an', 'is', 'to', 'be', 'am']
    
    for key, value in apos_dict.items():
        text = text.replace(key, value)

    # clean the text a bit 
    text = text.replace("\n", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("br", "")
    text = text.replace("\r", "")
    text = text.replace("/", "") 

    # replace the useless words 
    text_list = text.split()
    new_list = [] 
    for word in text_list: 
        if word not in useless: 
            new_list.append(word) 

    text = " ".join(new_list) 

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # score the words by occurrences only 
    word_dict = {} 
    for word in doc: 
        word = word.text.lower() 
        if word in word_dict:
            word_dict[word] += 1
        else: 
            word_dict[word] = 1
    
    sents = [] 
    sent_score = 0 
    for index, sent in enumerate(doc.sents):
        for word in sent: 
            word = word.text.lower() 
            sent_score += word_dict[word] 
        sents.append((sent.text.replace("\n", " "), sent_score/len(sent), index))  

    sents = sorted(sents, key=lambda x: -x[1])  
    sents = sorted(sents[:3], key=lambda x: x[2]) 

    summary = "" 
    for sent in sents: 
        summary += sent[0] + " "

    return summary 

# function to generate wordcloud for different text categories 
def generate_wordcloud(category_list):
    text_list = [str(elem) for elem in category_list]
    text = " ".join(text_list)
    text = text.replace("br", "")

    word_cloud = WordCloud(collocations=False, background_color='white').generate(text) 

    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show() 

# function to calculate average ratings when a keyword is used 
def get_keyword_average_rating(word, data):
    description_reviews = [] 
    about_reviews = [] 
    neighborhood_reviews = [] 

    for index in data.index:
        description = data['description'][index]
        about = data['host_about'][index]
        overview = data['neighborhood_overview'][index]
        review = data['review_scores_rating'][index]
        if word in str(description):
            description_reviews.append(review) 
        if word in str(about):
            about_reviews.append(review) 
        if word in str(overview):
            neighborhood_reviews.append(review) 

    desc_rating = statistics.mean(description_reviews)
    about_rating = statistics.mean(about_reviews)
    hood_rating = statistics.mean(neighborhood_reviews)

    return [desc_rating, about_rating, hood_rating]

# generate wordclouds 
generate_wordcloud(data['neighborhood_overview'])
generate_wordcloud(data['description'])
generate_wordcloud(data['host_about'])

avg_rating = statistics.mean(data['review_scores_rating'])

# plot ratings when "beach" is used 
beach_ratings = get_keyword_average_rating('beach', data)
beach_ratings.append(avg_rating)
plt.bar(['description', 'about', 'overview', 'average'], beach_ratings)
plt.title("Ratings when 'beach' is included")
plt.xlabel('description types')
plt.ylabel('ratings')
plt.show() 

# plot ratings when "Waikiki" is used
waikiki_ratings = get_keyword_average_rating('Waikiki', data) 
waikiki_ratings.append(avg_rating)
plt.bar(['description', 'about', 'overview', 'average'], waikiki_ratings)
plt.title("Ratings when 'Waikiki' is included")
plt.xlabel('description types') 
plt.ylabel('ratings')
plt.show() 

# summarize a random sampling of data 
random_data = data.sample(frac=0.0025)
desc_summaries = [] 
neighborhood_summaries = [] 
host_summaries = [] 
for elem in random_data['description']:
    desc_summaries.append(text_summarizer(str(elem)))

for elem in random_data['neighborhood_overview']:
    neighborhood_summaries.append(text_summarizer(str(elem)))

for elem in random_data['host_about']:
    host_summaries.append(text_summarizer(str(elem)))

print(desc_summaries[0], neighborhood_summaries[0], host_summaries[0])
