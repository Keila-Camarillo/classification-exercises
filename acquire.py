import pandas as pd 
import env 
import os 

'''
1. Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. 
Obtain your data from the Codeup Data Science Database.
'''
def new_titanic_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connect_url to mySQL
    - return a df of the given query from the titanic_db
    """
    url = env.get_db_url('titanic_db')
    
    return pd.read_sql(SQL_query, url)


def get_titanic_data(directory, filename="titanic.csv"):
    """
    This function will:s
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df of the SQL_query
        - write df to csv
    - Output titanic df
"""
    SQL_query = "select * from passengers"
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename) 
        return df
    
    else:
        df = new_titanic_data(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df

'''
2. Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame. 
The returned data frame should include the actual name of the species in addition to the species_ids. Obtain your data from the Codeup Data Science Database.
'''
def new_iris_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connect_url to mySQL
    - return a df of the given query from the iris_db
    """
    url = env.get_db_url('iris_db')
    
    return pd.read_sql(SQL_query, url)

def get_iris_data(directory, filename="iris_db.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df of the SQL_query
        - write df to csv
    - Output iris df
"""
    SQL_query = "select * from measurements join species using(species_id)"
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename) 
        return df
    
    else:
        df = new_iris_data(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df

'''
Make a function named get_telco_data that returns the data from the telco_churn database in SQL. 
In your SQL, be sure to join contract_types, internet_service_types, payment_types tables with the customers table, so that the resulting dataframe 
contains all the contract, payment, and internet service options. Obtain your data from the Codeup Data Science Database.
'''
def new_telco_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connect_url to mySQL
    - return a df of the given query from the telco_churn
    """
    url = env.get_db_url('telco_churn')
    
    return pd.read_sql(SQL_query, url)

def get_telco_data(directory, filename="telco_churn.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df of the SQL_query
        - write df to csv
    - Output telco_churn df
"""
    SQL_query = ''' select * from customers
left join customer_churn
using (customer_id)
left join customer_signups
using (customer_id)
join internet_service_types
using (internet_service_type_id)
join payment_types
using (payment_type_id);'''

    if os.path.exists(directory + filename):
        df = pd.read_csv(filename) 
        return df
    
    else:
        df = new_telco_data(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df