
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from gensim import parsing # To stem data
from joblib import dump, load #save or load model
from sklearn.metrics import accuracy_score
from pdftotext import convert_pdf_to_string
from flask import jsonify

def train_model():
    try:
        #Read csv into a dataframe
        df = pd.read_csv("dataset/blooms_taxonomy_format_2.csv")
        #print first 5 rows of dataset
        print(df.head())

        # Any results you write to the current directory are saved as output.

        # for grouping similar words such as 'trying" and "try" are same words
        def parse(s):
            parsing.stem_text(s)
            return s


        # applying parsing to comments.
        for i in range(0, len(df)):
            df.iloc[i, 1] = parse(df.iloc[i, 1])

        # Seperate data into keyword and taxonomy level
        X, y = df['word'], df['taxonomy']

        # Split data in train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Use pipeline to carry out steps in sequence with a single object
        # SVM's rbf kernel gives highest accuracy in this classification problem.
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

        # train model
        text_clf.fit(X_train, y_train)

        dump(text_clf,'model.joblib') #save trained data into model

        return jsonify(success='Model has been created and saved')

    except Exception as e:
        print(e)
        return e


def predict(file_name):
    try:
        text_clf = load('model.joblib')
        pdf_string = convert_pdf_to_string(file_name)

        # input = pdf_string
        # input = [input]
        # predict class form test data

        predicted = text_clf.predict(pdf_string)
        predicted_list = tuple(set(predicted))
        return jsonify(blooms_taxonomy_levels=predicted_list)

    except Exception as e:
        print(e)
        return e

def all(file_name):
    try:
        #Read csv into a dataframe
        df = pd.read_csv("dataset/blooms_taxonomy_format_2.csv")
        #print first 5 rows of dataset
        print(df.head())

        # Any results you write to the current directory are saved as output.

        # for grouping similar words such as 'trying" and "try" are same words
        def parse(s):
            parsing.stem_text(s)
            return s


        # applying parsing to comments.
        for i in range(0, len(df)):
            df.iloc[i, 1] = parse(df.iloc[i, 1])

        # Seperate data into feature and results
        X, y = df['word'], df['taxonomy']

        # Split data in train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Use pipeline to carry out steps in sequence with a single object
        # SVM's rbf kernel gives highest accuracy in this classification problem.
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

        # train model
        text_clf.fit(X_train, y_train)

        dump(text_clf,'model.joblib') #save trained data into model
        pdf_string = convert_pdf_to_string(file_name)

        # input = pdf_string
        # input = [input]
        # print('converted_pdf',pdf_string)

        # predict class form test data
        predicted = text_clf.predict(pdf_string)
        predicted_list = tuple(set(predicted))
        return jsonify(blooms_taxonomy_levels=predicted_list)

    except Exception as e:
        print(e)
        return e