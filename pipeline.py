import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin


pd.set_option('display.width', None)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'AgeGroup']
num_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


class GeneralCleaner(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        age_bins = [0, 12, 18, 25, 30, 50, float('inf')]
        age_labels = ['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+']
        x['AgeGroup'] = pd.cut(x['Age'], bins=age_bins, labels=age_labels, right=False)
        del x['PassengerId']
        del x['Cabin']
        del x['Name']
        return x


numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessing_pipeline = ColumnTransformer(transformers=[
    ('numeric', numeric_pipeline, num_feats),
    ('categorical', categorical_pipeline, cat_feats)
])

full_pipeline = Pipeline([
    ('cleaner', GeneralCleaner()),
    ('preprocessing', preprocessing_pipeline),
    ('classifier', RandomForestClassifier())
])


def fit_model(train_data):
    x = train_data.copy()
    y = x.pop('Transported')
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=704)
    full_pipeline.fit(x_train, y_train)
    y_pred = full_pipeline.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f'Accuracy of the model is: {round(accuracy, 2)}')


def save_predictions_to_csv(model, x_test, output_file):
    y_test = pd.DataFrame({'PassengerId': x_test['PassengerId']})
    y_test['Transported'] = model.predict(x_test)
    print(y_test)
    # y_test.to_csv(output_file, index=False)


# fit_model(train)
# save_predictions_to_csv(full_pipeline, test, 'predictions/output_3_features.csv')


