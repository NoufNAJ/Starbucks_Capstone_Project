import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# read in the json files
portfolio = pd.read_json('Data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('Data/profile.json', orient='records', lines=True)
transcript = pd.read_json('Data/transcript.json', orient='records', lines=True)

# 1- Cleaning portfolio

# Create dummy variables for the 'Channels' column
channel_dummies = pd.get_dummies(portfolio.channels.apply(pd.Series).stack()).groupby(level=0).sum()

# Concatenate the original transcriptFrame with the dummy variables
portfolio = pd.concat([portfolio, channel_dummies], axis=1)

# Drop the original 'Channels' column
portfolio.drop('channels', axis=1, inplace=True)

# Apply one-hot encoding to the 'offer_type' column
portfolio = pd.get_dummies(portfolio, columns=['offer_type'])

# Convert from boolean to binary
portfolio['offer_type_bogo'] = portfolio['offer_type_bogo'].astype(int)
portfolio['offer_type_discount'] = portfolio['offer_type_discount'].astype(int)
portfolio['offer_type_informational'] = portfolio['offer_type_informational'].astype(int)

# Rename the 'id' column to 'offer_id' for clarity.
portfolio = portfolio.rename(columns={'id':'offer_id'})

# 2- Cleaning profile

# Rename the 'id' column to 'user_id' for clarity.
profile = profile.rename(columns={'id':'user_id'})

# Drop rows where age is 118
profile = profile[profile['age'] != 118]

# Replace missing income values with mean income
profile['income'] = profile['income'].fillna(profile['income'].mean())

# Remove Others from gender 
profile = profile[profile['gender'] != 'O']

# Replace missing gender values with mode (most frequent gender)
profile['gender'] = profile['gender'].fillna(profile['gender'].mode()[0])

# Map gender to binary values
gender_mapping = {'M': 1, 'F': 0 }
profile['gender'] = profile['gender'].map(gender_mapping)


# 3- Cleaning transcript


# Rename the 'person' column to 'user_id'
transcript = transcript.rename(columns={'person':'user_id'})

# Replace spaces in the 'event' column with underscores
transcript['event'] = transcript['event'].str.replace(' ', '_')

# Extract offer_id from value column
transcript['offer_id'] = [next(iter(i.values())) if next(iter(i.keys())) in ['offer id', 'offer_id'] else None for i in transcript.value]

# Extract and round amount from value column
transcript['amount'] = [np.round(next(iter(i.values())), decimals=2) if next(iter(i.keys())) == 'amount' else None for i in transcript.value]

# Extract 'amount' from value column
transcript['amount'] = transcript['value'].apply(lambda x: x.get('amount', None))

# Impute amount column with mean for missing values
transcript['amount'] = transcript['amount'].fillna(transcript['amount'].mean())


# Assign 0 in missing values of 'offer_id'
transcript['offer_id'] = transcript['offer_id'].fillna(0)

# Drop original value column
transcript = transcript.drop(columns='value')

# Encode event 
gender_mapping = {'offer_received': 0, 'offer_viewed': 1, 'transaction': 2, 'offer_completed' : 3 }
transcript['event'] = transcript['event'].map(gender_mapping)

# Merge dataframes
merged = pd.merge(profile,transcript, on='user_id')
merged_df = pd.merge(merged, portfolio, on='offer_id', how='right')


# Build Machine Learning Model 

# Define feature columns and target variable
features = ['gender','age','income','time','amount','reward','difficulty','duration','email','mobile','social','web',
            'offer_type_bogo','offer_type_discount','offer_type_informational']
target = 'event'

# Separate features and target
X = merged_df[features]
y = merged_df[target]

# Convert all column names to strings
X.columns = X.columns.astype(str)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Identify categorical and numerical columns
categorical_cols = ['gender', 'email', 'mobile', 'social', 'web', 'offer_type_bogo', 'offer_type_discount', 'offer_type_informational']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = StandardScaler()

# Transform numerical data
X_train_numerical = numerical_transformer.fit_transform(X_train[numerical_cols])
X_test_numerical = numerical_transformer.transform(X_test[numerical_cols])

# Convert transformed numerical data back to DataFrame with appropriate column names
X_train_numerical = pd.DataFrame(X_train_numerical, columns=numerical_cols, index=X_train.index)
X_test_numerical = pd.DataFrame(X_test_numerical, columns=numerical_cols, index=X_test.index)

# Combine numerical and categorical data
X_train_scaled = pd.concat([X_train_numerical, X_train[categorical_cols]], axis=1)
X_test_scaled = pd.concat([X_test_numerical, X_test[categorical_cols]], axis=1)

# Define the model
model = RandomForestClassifier(
    n_estimators=300,            # Use 300 estimators
    random_state=42,             # Set random state for reproducibility
    bootstrap=True,              # Use bootstrap sampling
    max_depth=None,              # No maximum depth
    min_samples_leaf=4,          # Minimum number of samples required to be at a leaf node
    min_samples_split=10,        # Minimum number of samples required to split an internal node
    n_jobs=-1                    # Use all available cores
)

model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)


# Train your model (as you did in your code)
model.fit(X_train_scaled, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(numerical_transformer, scaler_file)

print('Model Saved')