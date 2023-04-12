### IMPORTS ####
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import env

################

### FUNCTIONS ###

def acquire():
    database = 'mall_customers'
    url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
    df = pd.read_sql('SELECT * FROM customers', url, index_col='customer_id')
    return df

def one_hot_encode(df):
    df['is_female'] = df.gender == 'Female'
    df = df.drop(columns='gender')
    return df

def split(df):
    train_and_validate, test = train_test_split(df, random_state=13, test_size=.15)
    train, validate = train_test_split(train_and_validate, random_state=13, test_size=.2)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)

    return train, validate, test

def scale(train, validate, test):
    columns_to_scale = ['age', 'spending_score', 'annual_income']
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    return scaler, train_scaled, validate_scaled, test_scaled

def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print()
    print('--- Info')
    df.info()
    print()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def handle_outliers(df, cols, k):
    # Create placeholder dictionary for each columns bounds
    bounds_dict = {}

    # get a list of all columns that are not object type
    non_object_cols = df.dtypes[df.dtypes != 'object'].index


    for col in non_object_cols:
        # get necessary iqr values
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr

        #store values in a dictionary referencable by the column name
        #and specific bound
        bounds_dict[col] = {}
        bounds_dict[col]['upper_bound'] = upper_bound
        bounds_dict[col]['lower_bound'] = lower_bound

    for col in non_object_cols:
        #retrieve bounds
        col_upper_bound = bounds_dict[col]['upper_bound']
        col_lower_bound = bounds_dict[col]['lower_bound']

        #remove rows with an outlier in that column
        df = df[(df[col] < col_upper_bound) & (df[col] > col_lower_bound)]
    
    return df

def handle_missing_values(df, prop_required_column, prop_required_row):
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df

def get_exploration_data():
    df = acquire()
    
    print('Before dropping nulls, %d rows, %d cols' % df.shape)
    df = handle_missing_values(df, prop_required_column=.5, prop_required_row=.5)
    print('After dropping nulls, %d rows, %d cols' % df.shape)
    
    train, validate, test = split(df)
    
    return train

def get_modeling_data(scale_data=False):
    df = acquire()
    
    print('Before dropping nulls, %d rows, %d cols' % df.shape)
    df = handle_missing_values(df, prop_required_column=.5, prop_required_row=.5)
    print('After dropping nulls, %d rows, %d cols' % df.shape)
    
    print()
    
    print('Before removing outliers, %d rows, %d cols' % df.shape)
    df = handle_outliers(df, ['age', 'spending_score', 'annual_income'], 1.5)
    print('After dropping outliers, %d rows, %d cols' % df.shape)
    
    print()
    
    df = one_hot_encode(df)
    
    train, validate, test = split(df)
    
    if scale_data:
        return scale(train, validate, test)
    else:
        return train, validate, test
    
#################