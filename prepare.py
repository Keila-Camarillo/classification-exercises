import acquire as acq
import os 
import pandas as pd
from sklearn.model_selection import train_test_split

# Prep iris_df

def prep_iris(df):
    '''
    The function will clean the iris dataset
    '''
    df["species"] = df.species_name
    df = df.drop(axis=1, columns=["species_id", "measurement_id", "species_name"])
    dummy_df = pd.get_dummies(df[["species"]], drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

# get and prep iris_df

def get_prep_iris(directory=os.getcwd()):
    '''
    The function will get and prepare the iris dataset 
    Takes in one argument (directory=os.getcwd())
    '''
    df = acq.get_iris_data(directory)
    df = pd.DataFrame(df)
    # df.rename(columns={"species_name" : "species"})
    df["species"] = df.species_name
    df = df.drop(axis=1, columns=["species_id", "measurement_id", "species_name"])
    dummy_df = pd.get_dummies(df[["species"]], drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

# get, prep and split data

# will create, prep and split data 
def get_prep_split_iris(df=get_prep_iris(), stratify_col= "species"):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    train, validate, test = (df=get_prep_titanic(), stratify_col= "species")
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[stratify_col])
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[stratify_col])
    return train, validate, test

# Prep titanic 
def prep_titanic(df):
    '''
    This function will clean the the titanic dataset
    '''
    df = df.drop(columns =['embark_town','class','age','deck'])
    

    df.embarked = df.embarked.fillna(value='S')

    dummy_df = pd.get_dummies(df[['sex','embarked']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

# Create and prep titanic df
def get_prep_titanic(directory=os.getcwd()):
    '''
    This function will get and prepare the the titanic dataset
    '''
    df = acq.get_titanic_data(directory)
    df = pd.DataFrame(df)
    df = df.drop(columns =['embark_town','class','deck'])

    df.embarked = df.embarked.fillna(value='S')

    dummy_df = pd.get_dummies(df[['sex','embarked']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

# will create, prep and split data 
def get_prep_split_titanic(df=get_prep_titanic(), stratify_col= "survived"):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    train, validate, test = (df=get_prep_titanic(), stratify_col= "survived")
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[stratify_col])
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[stratify_col])
    return train, validate, test


#prep telco 
def prep_telco(df=acq.get_telco_data(directory=os.getcwd())):
    '''
    The function will clean the telco dataset
    '''
    # drop unecessary columns
    df = df.drop(columns=["customer_id","payment_type_id", "internet_service_type_id", "contract_type_id", "phone_service", "multiple_lines", "online_security", "online_backup", "device_protection", "streaming_tv", "streaming_movies"])
    # remove nulls and replace nulls with 0 (non-churn customers)
    df["churn_month"] = df.churn_month.fillna(0)
    df["total_charges"] = df.total_charges.replace(" ", 0).astype(float)
    # create dummies
    dummy_df = pd.get_dummies(df[["gender", "partner", "dependents", "paperless_billing", "internet_service_type", "payment_type", "tech_support", "churn"]], drop_first=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
    df = df.drop(columns=["gender", "partner", "dependents", "paperless_billing", "churn_month",  "internet_service_type", "payment_type", "tech_support", "signup_date", "churn"])
    df = pd.concat([df, dummy_df], axis=1)
    return df

#prep telco 
def get_prep_telco(directory=os.getcwd()):
    '''
    The function will clean the telco dataset
    '''
    df = acq.get_telco_data(directory)
    df = pd.DataFrame(df)
    # drop unecessary columns
    df = prep_telco(df) 
    return df

def get_prep_split_telco(df=prep_telco(), stratify_col= "churn_Yes"):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    train, validate, test = (df=get_prep_titanic(), stratify_col= "churn")
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[stratify_col])
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[stratify_col])
    return train, validate, test
# split data
def split_data(df, stratify_col):
    '''
    Takes in two arguments the dataframe name and the ("stratify_name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[stratify_col])
    train, validate = train_test_split(train, #second split
                                    test_size=.25, 
                                    random_state=123, 
                                    stratify=train[stratify_col])
    return train, validate, test