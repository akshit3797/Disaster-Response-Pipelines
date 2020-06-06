import sys
# import libraries
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import joblib


def load_data(database_filepath):
    '''
    Loads X (features) and Y (labels) and gets category names.
    
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data, just the messages
        y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages',engine)
    X = df.message
    y = df.iloc[:,4:]
    
    category_names = y.columns.tolist()
    
    return X,y, category_names


def tokenize(text):
    
    '''
    Basic tokenizer that do lower case, removes punctuations, stopwords and tokenizes then lemmatizes
    
    Args:
        text (string): input message to tokenize
    Returns:
        tokens (list): list of cleaned tokens in the message
        
    '''
        
    # Remove Punctuations
    text = re.sub(r"[^A-Za-z0-9]",' ',text)
    
    # Tokenize Sentence.
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(token,'v').lower().strip()
        clean_tokens.append(clean_tok)
    
    # StopWords Removal
    stop_words = set(stopwords.words('english'))
    clean_tokens = [token for token in clean_tokens if token not in stop_words]
    
    return clean_tokens


def build_model():
    
    '''
    Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    '''
    
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    
    parameters = {
    "tfidf__ngram_range":[(1,1),(1,2)],
    "tfidf__max_df":[0.5,0.75,1],
    "clf__estimator__penalty":['l1','l2'],
    "clf__estimator__C":[0.001,0.01,0.1,1],
    }

    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """
    
    y_pred = model.predict(X_test)
    
    print('Accuracy: ', accuracy_score(Y_test, y_pred))
    print('Precision: ', precision_score(Y_test, y_pred, average='micro'))
    print('Recall: ', recall_score(Y_test, y_pred, average='micro'))
    print('F1 Score: ',f1_score(Y_test, y_pred, average='micro'))

    # Print out the full classification report
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    '''
    
    filename = model_filepath
    joblib.dump(model, filename)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
        # Since child_alone does not have any data related to it, we remove it
        Y_train = Y_train.drop(columns='child_alone',axis=1)
        Y_test = Y_test.drop(columns='child_alone', axis=1)
        category_names.remove('child_alone')
        
          
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')#3919

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()