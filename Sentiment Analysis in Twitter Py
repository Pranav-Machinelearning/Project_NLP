import pandas as pd
#get data from dataset
semantic_data = pd.read_csv(r"C:\Users\prana\OneDrive\Documents\Dataset\Semantic analysis.csv", names=["ID", "Hashtags", "Message status", "Twitter Messages"])
print(semantic_data)

# print(semantic_data.shape)
print(semantic_data.info())

semantic_data['Message status'] = semantic_data['Message status'].map({'negative': 0, 'positive': 1})

print(semantic_data.head(20))


figure = plt.figure(figsize=(12, 10))
semantic_data['Message status'].value_counts(normalize=True).plot(kind='bar', color=['#a4c639', '#5d8aa8'], alpha=0.9, rot=0)
plt.title('Positive & Negative: Unbalanced Dataset with No(0) and Yes(1) Indicators')
plt.show()

from sklearn.utils import resample

no_r = semantic_data[semantic_data['Message status'] == 0]
yes_r = semantic_data[semantic_data['Message status'] == 1]

# Oversample the minority class ('yes') to balance the dataset
re = resample(yes_r, replace=True, n_samples=len(no_r), random_state=123)

# Concatenate the oversampled 'yes' data with the 'no' data
overspl = pd.concat([no_r, re])

plt_figure = plt.figure(figsize=(12, 10))
overspl['Message status'].value_counts(normalize=True).plot(kind='bar', color=['#a4c639', '#5d8aa8'], alpha=1.0, rot=0)
plt.title('Balanced dataset After oversampling, Twitter semantics with negative and positive shows No(0) and Yes(1)')
plt.show()

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}



## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


import re
import pandas as pd
from nltk.stem import WordNetLemmatizer

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'
alphaPattern = "[^a-zA-Z0-9]"
sequencePattern = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

wordLemm = WordNetLemmatizer()

def preprocesstweet(textdata):
    tweettext = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        tweettext.append(tweetwords)
        
    return tweettext

text = overspl['Twitter Messages'].tolist()

t = time.time()
tweettext = preprocesstweet(text)
print('Text preprocessing complete.')
print(f'Time Taken: {round(time.time() - t)} seconds')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the preprocessed tweet texts
tfidf_features = tfidf_vectorizer.fit_transform(tweettext)

print(f'TF-IDF Vectorizer fit and transformed the data.')

import time
t = time.time()
tweettext = preprocesstweet(text)
print(f'Text preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')

negative_data = tweettext[:6000]
plt.figure(figsize = (20,20))
wd = WordCloud(max_words = 2000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(negative_data))
plt.imshow(wd)

positive_data = tweettext[6000:]
wd = WordCloud(max_words = 2000 , width = 1600 , height = 800,
              collocations=False).generate(" ".join(positive_data))
plt.figure(figsize = (20,20))
plt.imshow(wd)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, overspl['Message status'], test_size=0.1, random_state=42)

def Evl(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    
    # Compute and plot the Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot = labels, cmap = 'Blues', fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()

# Initialize and train BernoulliNB model
BNBmodel = BernoulliNB(alpha=2)
BNBmodel.fit(X_train, y_train)
Evl(BNBmodel)

# Initialize and train LinearSVC model
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)

# Initialize and train LogisticRegression
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)

# Function to preprocess and predict sentiment of new tweets
def predict_sentiment(new_tweets, model, vectorizer):
    processed_tweets = preprocesstweet(new_tweets)
    tweet_features = vectorizer.transform(processed_tweets)
    predictions = model.predict(tweet_features)
    return predictions

# Example usage with new tweets
new_tweets = [
    "I love machine learning!",
    "This is the worst university.",
    "I have completed my assignment.",
    "The lecture was so boring!"
]

# Predict sentiments using the trained Logistic Regression model
predicted_sentiments = predict_sentiment(new_tweets, LRmodel, tfidf_vectorizer)

# Print predictions
for tweet, sentiment in zip(new_tweets, predicted_sentiments):
    sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
    print(f'Tweet: "{tweet}" - Sentiment: {sentiment_label}')
