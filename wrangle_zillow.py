'''Wrangles data from Zillow Database'''

##################################################Wrangle.py###################################################

import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import user, pwd, host
import warnings
warnings.filterwarnings('ignore')

#**************************************************Acquire*******************************************************

def acquire_zillow():
    ''' 
     Acquire data from Zillow using env imports, rename columns, and storing a cached version of SQL pull as a .csv.
     Specifically, the SQL query returns SINGLE-FAMILY property results which were the subject of a transaction in 2017.
     '''    
    
    
    
    
    
    if os.path.exists('zillow_pred_2017.csv'):
        print('local version found!')
        return pd.read_csv('zillow_pred_2017.csv', index_col=0)
    else:
       

        url = f"mysql+pymysql://{user}:{pwd}@{host}/zillow"

        query = """

        SELECT predictions_2017.logerror as logererror,bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, lotsizesquarefeet, fips,
                regionidcity
        FROM properties_2017

        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        
        JOIN predictions_2017 USING(parcelid)

        WHERE propertylandusedesc IN ("Single Family Residential",                       
                                      "Inferred Single Family Residential")"""

        # get dataframe of data
        df = pd.read_sql(query, url)


        # renaming column names to one's I like better
        df = df.rename(columns = {'bedroomcnt':'beds', 
                                  'bathroomcnt':'baths', 
                                  'calculatedfinishedsquarefeet':'sqft',
                                  'lotsizesquarefeet':'lotsqft',
                                  'taxvaluedollarcnt':'taxable_value', 
                                  'yearbuilt':'built',
                                 'regionidcity':'city'})
        df.to_csv('zillow_pred_2017.csv',index=True)
        
        return df

#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' Remove outliers from a list of columns in a dataframe 
        and return that dataframe.  This function takes in a dataframe, a multiplier applied to IQR used to define
        an outlier (recommend 1.5), and finally a list of columns subjected to the function.
        
        Outputs a dataframe with any rows containing outliers in any of the specified columns REMOVED from the
        input dataframe.
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips','city','beds','baths']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=50)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['sqft', 'taxable_value','built','lotsqft']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]],x=col)

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
        
#**************************************************Prepare*******************************************************

def prepare_zillow(df):
    ''' 
      Takes in a Zillow df      - removes outliers
                                - displays histograms and boxplots for cont and discrete data
                                - converts specific columns dtypes
                                - drops NULLS
                                - Creates train/val/test split
                                - Imputes missing Year_Built based upon training into val and test
    '''

    # removing outliers
    df = remove_outliers(df, 1.5, ['taxable_value','sqft','lotsqft'])
    df = df[(df['baths'] >= 1) & (df['baths'] <= 4)]
    df = df[(df['beds'] >= 1) & (df['beds'] <= 5)]
    
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)
    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.built = df.built.astype(object)
    df.city = df.city.astype(object)
    
    #remove nulls
    df = df.dropna()
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')
    # fit the imputer once on train
    imputer.fit(train[['built']])
    #impute (transform) the values learned from train on train, val, test
    train[['built']] = imputer.transform(train[['built']])
    validate[['built']] = imputer.transform(validate[['built']])
    test[['built']] = imputer.transform(test[['built']])   
    
    return train, validate, test    


#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test


def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in each column
    it returns a dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    percent_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')