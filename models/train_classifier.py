# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



def load_data(database_filepath):
    """ 
    Loads data from the created database 
    
    Inputs:
    database_filepath - String path to the CSV file containing messages
    
    Output:
    X - Dataframe containing the messages as input features
    y- Dataframe containing the categories as output/target variables
    category_names- column names of the categories
    """

    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Messages',engine)
    X =df['message'] 
    y =df.iloc[:,4:]  
    category_names=y.columns
    return X,y,category_names


def tokenize(text):
    """ 
    Returns cleaned tokens from the text provided" 
    
    Inputs:
    text- the text provided    
    
    Output:
    clean_tokens -list of cleaned tokens from the text provided
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
       
    text=re.sub(r"[^a-zA-Z0-9]"," ",text)

    tokens = word_tokenize(text)
    stopwords1=stopwords.words("english")
    tokens = [w for w in tokens if w not in stopwords1]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """ 
    Builds a Machine Learning Pipeline 
    
    Inputs:
    None
    
    Output:
    """
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])   
    parameters = {'clf__estimator__n_estimators':[100,150,200],
                  'clf__estimator__min_samples_split': [2,5,10]}
               

    cv = GridSearchCV(pipeline,param_grid=parameters,n_jobs=4,cv=3,verbose=2)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """ 
    Prints the classification metrics/performance of the model
    
    Inputs:
    model- Trained model
    X_test- Training set
    y_test- Training set labels/targets
    category_names- Column names for the targets
    
    Output:
    None
    """
    y_test_pred=model.predict(X_test)

    for i in range(len(category_names)):
    
        print(y_test.columns[i],'category metrics :')
        print(classification_report(y_test.iloc[:,i],y_test_pred[:,i]))


def save_model(model, model_filepath):
    """ 
    Saves the model to a pickle file
    
    Inputs:
    model: Trained Scikitlearn model
    model_filepath: Destination path to save the pickle file   
 
    Output:
    None
    """
    with open(model_filepath,'wb') as file:
        pickle.dump(model,file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=142)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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