import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
#% matplotlib inline

import os
flat_list = []

directory = 'cable_news_copy'

for filename in os.listdir(directory):
    with open(os.getcwd() + '/' + directory + '/' + filename) as f:
        try:
            flat_list = flat_list + [word for line in f for word in line.split()]
        except:
         continue

import nltk
import pandas as pd
from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

# init lemmatizer
lemmatizer = WordNetLemmatizer()
flat_list = [lemmatizer.lemmatize(word=word,pos='v') for word in flat_list]

tags = nltk.pos_tag(flat_list)

flat_list = [word for word, pos in tags if (pos == 'VERB' or pos == 'NN' or pos == 'NNP' or pos == "JJ")
]

flat_list = [word for word in flat_list if len(word) > 3]

stopwords = set(STOPWORDS)

stopwords.update(["people", "one", "comment", "think", "still", "really", "thing",
 "something", "someone", "right", "person", "going", "around", "every", "com", "www",
 "reddit", "comment", "gonna", "stuff", "thought", "comments", "deleted", "people",
 "coronavirus", "things", "everyone", "anything", "ll", "anyone", "re", "thank", "nothing", "01", "everything", "html",
 "point", "place", "x200b", "please", "chance", "well", "anche", "https", "subreddit",
 "happen", "000", "start", "question", "edit", 
 "call", "know", "tell", "much", "many", "stay", "make", "days", "good", "sure", "post",
"time", "real", "long", "kind", "fine", "whole", "now", "last", "idea", "true", "different", "news",
"information", "likely", "message", "hand", "first", "city", "corona", "week", "yeah", "doesn", "isn",
"nta", "ve", "come", "country", "covid", "19", "date", "york", "new", "body", "mercury", "california", "washington", "sacramento",
"times", "santa", "clara", "county", "south", "united", "northern", "seattle", "davis", "tuesday", "wednesday", "monday",
"thursday", "friday", "saturday", "sunday", "davis", "february", "march", "january", "year", "press", "document",
"reserved", "byline", "copyright", "ap", "section", "english" "publication", "associated", "percent", "said",
"eastern", "part", "president", "lemon", "carlson", "tucker", "gutfeld", "perino", "hannity", "don", "cooper", "yes",
"donald", "gupta", "john", "bret", "williams", "hayes", "baier", "look", "end", "way", "let", "today", "day", "vice", "tonight",
"house", "sort", "break", "issue", "unidentified", "crosstalk", "roberts", "great", "report", "male", "female", "watters", "kong",
"clip", "video", "ship", "everybody", "chief", "correspondent", "able", "next", "begin", "talk"
 ])

stopwords.update([str(num) for num in range(0,100)])


# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import seaborn as sns
sns.set_style('whitegrid')

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    print(words)
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = TfidfVectorizer(stop_words=stopwords)

count_data = count_vectorizer.fit_transform(flat_list)
# Visualise the 10 most common words

dic = count_vectorizer.get_feature_names()

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 3
number_words = 6
# Create and fit the LDA model

#%%time
from pyLDAvis import sklearn as sklearn_lda
import pickle, os
import pyLDAvis

#while number_topics <= 6:

lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)

model = (dic,lda.components_,lda.exp_dirichlet_component_,lda.doc_topic_prior_)

# Print the topics found by the LDA model
print("***************** {} Topics found via LDA: ******************".format(number_topics))
print_topics(lda, count_vectorizer, number_words)


LDAvis_data_filepath = os.path.join('./ldavis_prepared_cablenews_opinion_'+str(number_topics))

LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_cablenews_opinion_'+ str(number_topics) +'.html')


with open('outfile_cable_news', 'wb') as fp:
    pickle.dump(model, fp)

plot_10_most_common_words(count_data, count_vectorizer)