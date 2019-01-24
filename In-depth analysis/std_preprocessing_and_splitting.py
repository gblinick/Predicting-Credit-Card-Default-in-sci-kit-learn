
#### Step 1) Preprocess Data

# We will train our classifier with the following features:
# Numeric features to be scaled: LIMIT_BAL, AGE, PAY_X, BIL_AMTX, and PAY_AMTX
# Categorical features: SEX, EDUCATION, MARRIAGE

# We create the preprocessing pipelines for both numeric and categorical data
numeric_features = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                     'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
                   ]



data['PAY_1'] = data.PAY_1.astype('float64')
data['PAY_2'] = data.PAY_2.astype('float64')
data['PAY_3'] = data.PAY_3.astype('float64')
data['PAY_4'] = data.PAY_4.astype('float64')
data['PAY_5'] = data.PAY_5.astype('float64')
data['PAY_6'] = data.PAY_6.astype('float64')
data['AGE'] = data.AGE.astype('float64')

numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(categories='auto'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        #,('lab', label_transformer, label_features)
    ])
  
#### Step 2) Split Data into Training and Test Sets

y = data['default']#.values
X = data.drop(['default'], axis=1)#.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
    
    
    
    
    