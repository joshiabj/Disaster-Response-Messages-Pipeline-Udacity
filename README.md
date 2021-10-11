# Disaster Response Pipeline Project

### Introduction to the project:

This project looks at analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages.
The project comprises of a data set containing real messages that were sent during disaster events. 

The following analysis was carried out in the project:
1) A ETL pipeline is created to extract the data from the source, clean the data using different techniques and save it to a SQLite database.
2) A machine learning pipeline is created to categorize these events so that one can send the messages to an appropriate disaster relief agency.
3) A web app is also created using Flask where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Dependencies:

Python Version: 3.5+
Libraries Required:
1) Numpy
2) Scikit-Learn
3) Pandas 
4) NLTK
5) SQLalchemy
6) Pickle

Web Application and Visualisation: Flask and Plotly

### GitHub Link:
https://github.com/joshiabj/Disaster-Response-Messages-Pipeline-Udacity

### File Structure and description:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # ETL Pipeline
|- DisasterResponse.db# database to save clean data to
|- ML_Pipeline_Preparation # step by step process to build ML pipeline(useful for understanding ML pipeline)
|- ETL_pipeline_preparation # step by step process to build ETL pipeline(useful for understanding ETL pipeline)

- models
|- train_classifier.py # ML Pipeline
|- classifier_1.pkl  # saved model 

- README.md


### Instructions for running the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier_1.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Additional Details

1) The process.py file consists of the ETL pipeline. The transformation phase looks at cleaning the data by converting the categories variable to numerical and encoding them.
2) The train_classifier.py consists of the ML pipeline. The cleaning process for the loaded dataset looks at applying NLP techniques such as removing stop words, lemmatizing, removing URLs/punctuation for the messages variable.
3) Certain categories such as missing_person, security have a very high F1 score for non-presence(0) and a very low F1 score for presence(1). This is caused due to the class imbalance in the messages dataset.
4) The classifier_1.pkl pickle file contains the trained model.

