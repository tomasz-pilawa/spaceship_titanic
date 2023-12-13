import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.width', None)
pd.options.display.max_rows = None

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


class GeneralCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, make_ordinals=False):
        self.make_ordinals = make_ordinals

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):

        x[['Group', 'Member']] = x['PassengerId'].str.split('_', expand=True)
        gc = x['Group'].value_counts().sort_index()
        x['Travelling_Solo'] = x['Group'].apply(lambda fu: fu not in set(gc[gc > 1].index))
        x['GroupSize'] = x.groupby('Group')['Member'].transform('count')

        x[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = x['Cabin'].str.split('/', expand=True)

        x["Total_Expenditure"] = x[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        x["No_Spending"] = (x["Total_Expenditure"] == 0)

        exp_mean = round(x["Total_Expenditure"].mean(), 2)
        exp_median = x["Total_Expenditure"].median()
        x['Expenditure_Category'] = pd.cut(x['Total_Expenditure'],
                                           bins=[-1, 0, exp_median, exp_mean, float('inf')],
                                           labels=['No_Expense', 'Low_Expense', 'Medium_Expense', 'High_Expense'])

        x['Amenities_Used'] = x[['RoomService', 'Spa', 'VRDeck', 'FoodCourt', 'ShoppingMall']].gt(0).sum(axis=1).astype(
            float)
        x['Spending_Service'] = x[['RoomService', 'Spa', 'VRDeck']].sum(axis=1)
        x['Spending_Shopping'] = x[['FoodCourt', 'ShoppingMall']].sum(axis=1)
        x['Surname'] = x['Name'].str.split().str[-1]

        # Manual Imputation Parsed Here for Ease of Use

        x['HomePlanet'] = x.groupby('Group')['HomePlanet'].transform(
            lambda group: group.fillna(group.mode().iloc[0]) if not group.mode().empty else np.nan)

        cabin_to_homeplanet = {'A': 'Europa', 'B': 'Europa', 'C': 'Europa', 'T': 'Europa', 'G': 'Earth'}
        x['HomePlanet'] = x['Cabin_Deck'].map(cabin_to_homeplanet).fillna(x['HomePlanet'])

        x['HomePlanet'] = x.groupby('Surname')['HomePlanet'].transform(
            lambda g: g.fillna(g.mode().iloc[0]) if not g.mode().empty and not pd.isna(g.name) else g).fillna(
            x['HomePlanet'])

        x.loc[(x['Cabin_Deck'] == 'D') & x['HomePlanet'].isna(), 'HomePlanet'] = 'Mars'

        x['Surname'] = x.groupby('Group')['Surname'].transform(
            lambda t: t.fillna(t.mode()[0]) if not t.mode().empty else 'Unknown')

        x['Family_Size'] = x.groupby('Surname')['Surname'].transform('count')
        x.loc[x['Surname'] == 'Unknown', 'Surname'] = np.nan
        x.loc[x['Family_Size'] > 100, 'Family_Size'] = 0

        for n in ['Cabin_Deck', 'Cabin_Side']:
            x[n] = x.groupby('Group')[n].transform(
                lambda t: t.fillna(t.mode()[0]) if not t.mode().empty else np.nan)
            x[n] = x.groupby(['HomePlanet', 'Destination', 'Travelling_Solo'])[n].transform(
                lambda r: r.fillna(r.mode()[0]) if not r.mode().empty and not pd.isna(r.name) else r).fillna(x[n])

        for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            if x[(x['Cabin_Deck'] == deck) & (x['Cabin_Number'].isna())].shape[0] > 0:
                X_CN = x.loc[~(x['Cabin_Number'].isna()) & (x['Cabin_Deck'] == deck), 'Group']
                y_CN = x.loc[~(x['Cabin_Number'].isna()) & (x['Cabin_Deck'] == deck), 'Cabin_Number']
                X_test_CN = x.loc[(x['Cabin_Number'].isna()) & (x['Cabin_Deck'] == deck), 'Group']

                model_CN = LinearRegression()
                model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
                preds_CN = model_CN.predict(X_test_CN.values.reshape(-1, 1))
                x.loc[(x['Cabin_Number'].isna()) & (x['Cabin_Deck'] == deck), 'Cabin_Number'] = preds_CN.astype(int)

        x['Cabin_Number'].fillna(x['Cabin_Number'].median(), inplace=True)
        x['Cabin_Number'] = x['Cabin_Number'].astype(int)
        x.loc[x['Cabin_Number'] < 0, 'Cabin_Number'] = 0

        x['Cabin'] = pd.cut(x['Cabin_Number'], bins=[0, 300, 600, 900, 1200, 1500, float('inf')],
                            labels=['Region1', 'Region2', 'Region3', 'Region4', 'Region5', 'Region6'], right=False)

        cabin_encoder = LabelEncoder()
        x['Cabin'] = cabin_encoder.fit_transform(x['Cabin'])

        x.loc[x['Age'].isna(), 'Age'] = \
            x.groupby(['HomePlanet', 'No_Spending', 'Travelling_Solo', 'Cabin_Deck'])['Age'].transform(
                lambda a: a.fillna(a.median()))[x.loc[x['Age'].isna(), 'Age'].index]

        x['Age_Group'] = pd.cut(x['Age'], bins=[0, 12, 18, 25, 30, 50, float('inf')], right=False,
                                labels=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])

        x['CryoSleep'] = x.groupby('No_Spending')['CryoSleep'].transform(lambda cs: cs.fillna(cs.mode()[0]))

        if self.make_ordinals:
            age_group_encoder = LabelEncoder()
            exp_category_encoder = LabelEncoder()
            x['Age_Group'] = age_group_encoder.fit_transform(x['Age_Group'])
            x['Expenditure_Category'] = exp_category_encoder.fit_transform(x['Expenditure_Category'])

        del x['Age']
        del x['PassengerId']
        del x['Name']
        del x['Group']
        del x['Member']
        del x['Cabin_Number']
        del x['Surname']

        return x


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_type='numeric'):
        self.feature_type = feature_type

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if self.feature_type == 'numeric':
            num_cols = x.select_dtypes(include=['float', 'int']).columns.tolist()
            return x[num_cols]
        elif self.feature_type == 'category':
            cat_cols = x.select_dtypes(exclude=['float', 'int']).columns.tolist()
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

    def __init__(self, drop_first=False):
        self.x_train = None
        self.columns = []
        self.drop_first = drop_first

    def fit(self, x, y=None):
        self.x_train = pd.get_dummies(x, drop_first=self.drop_first)
        return self

    def transform(self, x, y=None):
        x_encoded = pd.get_dummies(x, drop_first=self.drop_first)
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
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.feature_union = FeatureUnion(transformer_list=self.transformer_list,
                                          n_jobs=self.n_jobs,
                                          transformer_weights=self.transformer_weights,
                                          verbose=self.verbose)

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

    def get_params(self, deep=True):
        return self.feature_union.get_params(deep=deep)


class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = []

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Expenditure',
                'Spending_Service', 'Spending_Shopping']
        for col in cols:
            x[col] = np.log1p(x[col])
        self.columns = x.columns
        return x

    def get_features_name(self):
        return list(self.columns)


cat_pipe = Pipeline(steps=[('cat_selector', FeatureSelector(feature_type='category')),
                           ('imputer', CustomImputer(strategy='most_frequent')),
                           ('encoder', CustomDummify())
                           ])

num_pipe = Pipeline(steps=[('num_selector', FeatureSelector(feature_type='numeric')),
                           ('imputer', CustomImputer(strategy='median')),
                           ('num_transformer', NumericTransformer())
                           ])

preprocessing = FeatureUnionCustom([('categorical', cat_pipe),
                                    ('numeric', num_pipe)
                                    ])

preview_df_pipe = Pipeline(steps=[('cleaner', GeneralCleaner()),
                                  ('preprocessor', preprocessing),
                                  ])

full_pipe = Pipeline(steps=[('cleaner', GeneralCleaner()),
                            ('preprocessor', preprocessing),
                            ('scaler', CustomScaler()),
                            ('model', RandomForestClassifier())
                            ])


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
    y_test.to_csv(output_file, index=False)


def grid_search(data, pipe, param_grid):
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                        cv=5, scoring='accuracy', n_jobs=-1, return_train_score=False)

    pd.options.mode.chained_assignment = None

    x = data.copy()
    y = x.pop('Transported')
    grid = grid.fit(x, y)

    pd.options.mode.chained_assignment = 'warn'

    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', ascending=False)

    return result, grid


models_initial = [RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(),
                  SVC(), GaussianNB(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(),
                  XGBClassifier(), CatBoostClassifier(), LGBMClassifier()]

models_final = [LGBMClassifier(), RandomForestClassifier(), XGBClassifier(), GradientBoostingClassifier()]

params = [
    {
        'model': [LGBMClassifier()],
        'model__n_estimators': [300, 500],
        'model__learning_rate': [0.01],
        'model__num_leaves': [13, 15, 19],
        'preprocessor__categorical__encoder__drop_first': [True],
        'scaler__method': ['standard'],
        'cleaner__make_ordinals': [False]
    },
    {
        'model': [RandomForestClassifier()],
        'model__n_estimators': [200, 300],
        'model__max_depth': [9, 10],
        'model__min_samples_split': [7, 8],
        'preprocessor__categorical__encoder__drop_first': [True],
        'scaler__method': ['standard'],
        'cleaner__make_ordinals': [False]
    },
    {
        'model': [XGBClassifier()],
        'model__n_estimators': [25, 50, 200],
        'model__learning_rate': [0.1, 0.3],
        'cleaner__make_ordinals': [False],
        'scaler__method': ['standard'],
        'preprocessor__categorical__encoder__drop_first': [False]
    }
]


def preview_df():
    df = preview_df_pipe.fit_transform(train)
    print(df)


def save_grid_predictions(csv_filename: str):
    preview_df()
    results, fitted_grid = grid_search(train, full_pipe, params)

    print(results)
    print(fitted_grid.best_params_)

    best_model = fitted_grid.best_estimator_
    save_predictions_to_csv(best_model, test, csv_filename)


# save_grid_predictions(csv_filename='predictions/output_17.csv')
