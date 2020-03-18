from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    dataset=pd.read_csv('C:/Users/Tarun Methwani/Desktop/nlp/data/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    corpus = []
    for i in range(0,1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).todense()
    y = dataset.iloc[:,1].values
    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.20, random_state = 0)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    classifier.score(X_test,y_test)
    # review_model = open("review_model.pkl","rb")
    # classifier = joblib.load(review_model)

    if request.method == 'POST':
        review =request.form['comment']
        new_review = re.sub("[^a-zA-Z]", " ", review)   
        new_review = new_review.lower().split()
        new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]   
        new_review = " ".join(new_review)
        new_review = [new_review]   
        new_review = cv.transform(new_review).toarray()
        my_prediction =classifier.predict(new_review)

    return render_template('result.html',prediction=my_prediction)

if __name__ =='__main':
    app.run(debug=True)