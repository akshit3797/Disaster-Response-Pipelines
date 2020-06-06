# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT : 
        1. messages_filepath : filepath of the messages dataset.
        2. categories_filepath : filepath of the categories dataset.
    RETURN : 
        It loads the data into two dataframes and merge them and returns them.
        
    '''
    
   
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merged dataframe on id column
    df = pd.merge(messages,categories,how='left',on='id')
    return df


def clean_data(df):
    '''
    Takes a Dataframe and perform cleaning operations such as expanding the multiple categories into seperate columns, extract     categories values, replace the previous categories with new columns and removing duplicates
    
    INPUT : 
        Merged dataframe
    
    RETURNS:
        cleaned dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # Split categories into seperate category columns
    category_colnames = categories.iloc[0].str[:-2].tolist()
       
    # rename the columns of `categories`
    categories.columns = category_colnames
    
#     print(categories)
    # Convert category values to just numbers 0 or 1. 
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
        
        
    # check number of duplicates
    print("No. of duplicates: ",df[df.duplicated()].shape[0])
        
    # drop duplicates
    df = df.drop_duplicates()
    print("Duplicates Removed.")
        
    # Drop missing values rows
    df.dropna(subset=category_colnames, inplace=True)
        
    # Replace category value from 2 to 0
    df['related'] = df['related'].replace({2:1})
        
    return df


def save_data(df, database_filename):
    '''
    Save the cleaned dataframe into the given database 
    
    INPUT: 
       1.  df: dataframe
       2.  database_filename: database to store the cleaned dataframe 
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False,if_exists='replace')

def main():
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