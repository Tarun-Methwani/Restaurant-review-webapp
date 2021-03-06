import numpy as np 
import pandas as pd 
import os
dataset = pd.read_csv('C:/Users/Tarun Methwani/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#CountVectorizer which converts the words in the dataset into 0 and 1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).todense()
y = dataset.iloc[:,1].values
#splitting the dataset into the training set and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
Accuracy_Score = accuracy_score(y_test, y_pred)

feedback = ""

newReview = ""

newReview = "worst"
def predict(new_review):   

        new_review = re.sub("[^a-zA-Z]", " ", new_review)   

        new_review = new_review.lower().split()

        new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]   

        new_review = " ".join(new_review)   

        new_review = [new_review]   

        new_review = cv.transform(new_review).toarray()   

        if classifier.predict(new_review)[0] == 1:

            return "Positive"   

        else:       

            return "Negative"

       

feedback = predict(newReview)

print("This review is: ", feedback) 
print("Accuracy Score is :", Accuracy_Score)