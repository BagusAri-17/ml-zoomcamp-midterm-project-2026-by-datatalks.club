# import library
import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

# standardize column names to snake_case
def cleaning_data(data):
    data = data.copy()
    data.columns = [re.sub(r'[^a-z0-9]+', '_', col.lower()).strip('_') for col in data.columns]
    return data

# load and preprocess dataset
def prepare_data(filepath='data/data.csv'):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df = cleaning_data(df)

    # filter only 'Dropout' and 'Graduate' (exclude active 'Enrolled' students)
    df = df[df['target'].isin(['Dropout', 'Graduate'])].reset_index(drop=True)

    # encode target as binary: Dropout = 1, Graduate = 0
    y = (df['target'] == 'Dropout').astype(int).values

    # extract features as dictionary records
    del df['target']
    dicts = df.to_dict(orient='records')
    return dicts, y

# vectorize feature dictionaries
def vectorize_data(dicts):
    print("Vectorizing features...")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    return dv, X

# train champion XGBoost model
def train_model(X, y):
    print("Training final XGBoost model...")
    model = XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=100,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)
    return model

# save model artifact and vectorizer
def save_model(dv, model, output_path='models/model.bin'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model successfully saved to {output_path}!")

# clean and concise main pipeline
def main():
    dicts, y = prepare_data('data/data.csv')
    dv, X = vectorize_data(dicts)
    model = train_model(X, y)
    save_model(dv, model, 'models/model.bin')

if __name__ == '__main__':
    main()
