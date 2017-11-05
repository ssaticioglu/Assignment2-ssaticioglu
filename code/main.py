import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train_tweets = np.load('./train_tweets.npy')
validation_tweets = np.load('./validation_tweets.npy')
train_classes = np.array([t[0] for t in train_tweets])
train_tweets = np.array([t[1] for t in train_tweets])
validation_classes = np.array([t[0] for t in validation_tweets])
validation_tweets = np.array([t[1] for t in validation_tweets])
#Bag of Words start
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', smooth_idf=True, decode_error='ignore')
X_train_idf = vectorizer.fit_transform(train_tweets)
#Training
clf = MultinomialNB().fit(X_train_idf, train_classes)
X_new_tfidf = vectorizer.transform(validation_tweets)
#Prediction
predicted = clf.predict(X_new_tfidf)
print accuracy_score(validation_classes, predicted)




