
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
#% matplotlib inline

#input data is in file titled reddit_posts.txt
with open('reddit_posts.txt') as f:
    flat_list=[word for line in f for word in line.split()]

import nltk
import pandas as pd
from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# init lemmatizer
lemmatizer = WordNetLemmatizer()
flat_list = [lemmatizer.lemmatize(word=word,pos='v') for word in flat_list]

tags = nltk.pos_tag(flat_list)

flat_list = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == "JJ"
    or pos == "VB" or pos == "CD")
]

stopwords = set(STOPWORDS)

stopwords.update(["people", "one", "comment", "think", "still", "really", "thing", "quando",
 "something", "someone", "right", "person", "going", "around", "every", "com", "www",
 "reddit", "comment", "gonna", "stuff", "thought", "comments", "deleted", "people",
 "coronavirus", "things", "everyone", "anything", "ll", "anyone", "re", "thank", "nothing", "qualcuno", "01", "everything", "html",
 "point", "place", "x200b", "please", "chance", "well", "anche", "https", "subreddit",
 "happen", "000", "start", "question", "edit", "questo", "della", "qualche", "fatto", "perche"])

new_list = []
for word in flat_list:
    word = word.lower()
    if word.startswith('don') == False and word not in stopwords and len(word) > 3:
        new_list.append(word)

flat_list = new_list

text = " ".join(word for word in flat_list if len(word) > 4)

print ("There are {} words in the combination of all review.".format(len(text)))

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer

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
count_vectorizer = CountVectorizer(stop_words=stopwords)
# Fit and transform the processed titles
flat_list = [word for word in flat_list if len(word) > 4]
flat_list = [word for word in flat_list if word not in stopwords]
count_data = count_vectorizer.fit_transform(flat_list)
# Visualise the 10 most common words
#plot_10_most_common_words(count_data, count_vectorizer)

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
#number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer)