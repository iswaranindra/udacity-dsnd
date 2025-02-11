import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle


""" 
    Load Data Function
  
    This is a function to load sqlite database.
  
    Parameters: 
    database_filepath (str): Path of where the data resides 
  
    Returns: 
    X (dataframe): Messages for prediction (feature)
    Y (dataframe): Categories of the given messages (target)
    category_names (list): List of categories
  
"""

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(np.array(Y.columns))

    return X,Y, category_names

""" 
    Tokenize Function
  
    This function convert messages into substring. Then being stemmed with 
    lemmatizer and finally its common morphological and inflexional endings
    got removed with PorterStemmer.
  
    Parameters: 
    text (str): Messages to be tokenized
  
    Returns: 
    clean_tokens (str): Cleaned messages
  
"""

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stemmer= PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = stemmer.stem(lemmatizer.lemmatize(tok).lower().strip())
        if clean_tok not in stopwords.words('english'): clean_tokens.append(clean_tok) 
    
    return clean_tokens

""" 
    Build Model Function
  
    This function executes pipeline which consist of tokenizer, 
    tfidf transformer, and Random Forest Classifier. Model then improved 
    using gridsearch with several parameters given.
  
    Parameters: 
    NA
  
    Returns: 
    cv (model): Trained model
  
"""

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__min_df': [1,2],
              'tfidf__use_idf':[True],
              'clf__estimator__n_estimators':[10,30], 
              'clf__estimator__min_samples_split':[2,5]
             }


    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

""" 
    Evaluate Model Function
  
    This function will print the performance of trained model 
    of predicting the test sets.
  
    Parameters: 
    model (model): Messages to be tokenized
    X_test (dataframe) : Test datasets of messages
    Y_test (dataframe) : Test datasets' categories
    category_names (list) : List of message categories
  
    Returns: 
    NA
  
"""

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
""" 
    Save Model Function
  
    This function will save the trained model in the path indicated.
  
    Parameters: 
    model (model): Trained model
    model_filepath (str) : Locations to save model
  
    Returns: 
    NA
  
"""

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

""" 
    Main Function - program entry point. 

"""

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()