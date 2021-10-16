#importing all the necessary libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 


def load_data(messages_filepath, categories_filepath):
    """ Loads messages data along with categories data
    
    Inputs:
    messages_filepath - String path to the CSV file containing messages
    categories_filepath- String path to the CSV file containing the different categories breakdown
    
    Output:
    df - Merged dataset containing both messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df
    
    
def clean_data(df):
    """ 
    Cleans the categories variable in the merged dataframe  
    
    Input:
    df - Merged dataset containing both messages and categories
    
    Output:
    df -  Dataset containing messages and the cleaned categories as columns
    """
    
    # creating a dataframe of the 36 individual category columns by splitting them
    categories = df['categories'].str.split(pat=";",expand=True)
    
    # Using the first row to extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = row.transform(lambda y: y[:-2])
    category_colnames=category_colnames.tolist()
    
    # renaming the columns of `categories`
    categories.columns = category_colnames
    
    #Cleaning the category columns to represent them as numbers 
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int) 
    
    #Removing the original uncleaned categories columns and merging with the dataframe
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],join='inner',axis=1)
    df.drop_duplicates(inplace=True)
    
    #Removing non binary values from the dataset
    df=df[df['related']!=2]
    return df


def save_data(df, database_filepath):
    """ 
     
     Saving the dataframe into a SQLite database
    
    Inputs:
    df - Cleaned dataset containing both messages and categories
    database_filepath - Filename for output sqlite database 
    
    Output:
    None
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql('Messages', engine, index=False,if_exists='replace')


def main():
    """
    The main function performs all the main data processing and cleaning functions such as
    1) Loading the messages and categories dataset and performing merge operations
    2) Cleaning the categories dataset and perfomring merge operations
    3) Saving the cleaned dataframe to SQLite database
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()