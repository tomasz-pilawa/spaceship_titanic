import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('display.width', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


class GeneralCleaner(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x['Age_Group'] = pd.cut(x['Age'], bins=[0, 12, 18, 25, 30, 50, float('inf')], right=False,
                                labels=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])

        x[['Group', 'Member']] = x['PassengerId'].str.split('_', expand=True)
        gc = x['Group'].value_counts().sort_index()
        x['TravellingSolo'] = x['Group'].apply(lambda fu: fu not in set(gc[gc > 1].index))
        x['GroupSize'] = x.groupby('Group')['Member'].transform('count')

        x[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = x['Cabin'].str.split('/', expand=True)
        x['Cabin_Number'].fillna(x['Cabin_Number'].median(), inplace=True)
        x['Cabin_Number'] = x['Cabin_Number'].astype(int)

        x['Cabin'] = pd.cut(x['Cabin_Number'], bins=[0, 300, 600, 900, 1200, 1500, float('inf')],
                            labels=['Region1', 'Region2', 'Region3', 'Region4', 'Region5', 'Region6'], right=False)

        x["Total_Expenditure"] = x[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        x["No_Spending"] = (x["Total_Expenditure"] == 0)

        exp_mean = round(x["Total_Expenditure"].mean(), 2)
        exp_median = x["Total_Expenditure"].median()
        x['Expenditure_Category'] = pd.cut(x['Total_Expenditure'],
                                           bins=[-1, 0, exp_median, exp_mean, float('inf')],
                                           labels=['No_Expense', 'Low_Expense', 'Medium_Expense', 'High_Expense'])

        del x['Age']
        del x['PassengerId']
        del x['Name']

        del x['Group']
        del x['Member']
        del x['Cabin_Number']

        return x


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_type='numeric'):
        self.feature_type = feature_type

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if self.feature_type == 'numeric':
            num_cols = x.columns[x.dtypes == float].tolist()
            return x[num_cols]
        elif self.feature_type == 'category':
            cat_cols = x.columns[x.dtypes != float].tolist()
            return x[cat_cols]


class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None
        self.columns = []

    def fit(self, x, y=None):
        self.imp = SimpleImputer(strategy=self.strategy)
        self.imp.fit(x)
        self.statistics_ = pd.Series(self.imp.statistics_, index=x.columns)
        return self

    def transform(self, x, y=None):
        x_imp = self.imp.transform(x)
        x_pd = pd.DataFrame(x_imp, index=x.index, columns=x.columns)
        self.columns = x_pd.columns
        return x_pd

    def get_features_name(self):
        return list(self.columns)


class CustomDummify(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.x_train = None
        self.columns = []

    def fit(self, x, y=None):
        self.x_train = pd.get_dummies(x)
        return self

    def transform(self, x, y=None):
        x_encoded = pd.get_dummies(x)
        missing_cols = set(self.x_train.columns) - set(x_encoded.columns)
        for col in missing_cols:
            x_encoded[col] = 0
        x_encoded = x_encoded[self.x_train.columns]
        self.columns = x_encoded.columns
        return x_encoded

    def get_features_name(self):
        return list(self.columns)


class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, method='standard'):
        self.scl = None
        self.method = method

    def fit(self, x, y=None):
        if self.method == 'standard':
            self.scl = StandardScaler()
        elif self.method == 'robust':
            self.scl = RobustScaler()
        self.scl.fit(x)
        return self

    def transform(self, x, y=None):
        x_scl = self.scl.transform(x)
        x_scaled = pd.DataFrame(x_scl, index=x.index, columns=x.columns)
        return x_scaled


class FeatureUnionCustom(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_list, verbose=False):
        self.transformer_list = transformer_list
        # self.verbose = verbose
        self.feature_union = FeatureUnion(transformer_list)

    def fit(self, x, y=None):
        self.feature_union.fit(x)
        return self

    def transform(self, x, y=None):
        x_trf = self.feature_union.transform(x)

        columns = []
        for transformer in self.transformer_list:
            cols = transformer[1].steps[-1][1].get_features_name()
            columns += list(cols)
        x_transformed = pd.DataFrame(x_trf, index=x.index, columns=columns)
        return x_transformed


# class NumericTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.columns = []
#
#     def fit(self, x, y=None):
#         return self
#
#     def transform(self, x, y=None):
#         cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total Expenditure']
#         for col in cols:
#             x[col] = np.log1p(x[col])


cat_pipe = Pipeline(steps=[('cat_selector', FeatureSelector(feature_type='category')),
                           ('imputer', CustomImputer(strategy='most_frequent')),
                           ('encoder', CustomDummify())
                           ])

num_pipe = Pipeline(steps=[('num_selector', FeatureSelector(feature_type='numeric')),
                           ('imputer', CustomImputer(strategy='median')),
                           ])

preprocessing = FeatureUnionCustom([('categorical', cat_pipe),
                                    ('numeric', num_pipe)
                                    ])

full_pipe = Pipeline(steps=[('cleaner', GeneralCleaner()),
                            ('preprocessor', preprocessing),
                            ('scaler', CustomScaler()),
                            ('model', RandomForestClassifier())
                            ])

# TESTING THE PIPE DF OUTPUT - COMMENT 'model' IN full_pipe
# df = full_pipe.fit_transform(train)
# print(df)


def fit_model(train_data):
    x = train_data.copy()
    y = x.pop('Transported')
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=704)
    full_pipe.fit(x_train, y_train)
    y_pred = full_pipe.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f'Accuracy of the model is: {round(accuracy, 2)}')


def save_predictions_to_csv(model, x_test, output_file):
    y_test = pd.DataFrame({'PassengerId': x_test['PassengerId']})
    y_test['Transported'] = model.predict(x_test)
    print(y_test)
    y_test.to_csv(output_file, index=False)


fit_model(train)
save_predictions_to_csv(full_pipe, test, 'predictions/output_7_total_exp.csv')
