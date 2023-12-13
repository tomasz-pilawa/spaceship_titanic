import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

pd.set_option('display.width', None)


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


def baseline_processing(data):

    x_processed = data.copy()

    if 'Transported' in x_processed.columns:
        y_processed = x_processed.pop('Transported')
    else:
        y_processed = pd.DataFrame({'PassengerId': x_processed['PassengerId'], 'Transported': None})

    x_processed = pd.get_dummies(x_processed, columns=['HomePlanet', 'Destination', 'VIP'])
    x_processed = x_processed.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

    imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed
    x_processed = pd.DataFrame(imputer.fit_transform(x_processed), columns=x_processed.columns)

    return x_processed, y_processed


def baseline_model(data):
    x, y = baseline_processing(data)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    print(f'Accuracy of the baseline model is: {round(accuracy, 2)}')

    return model


def get_baseline_predictions(data_train, data_test):
    model = baseline_model(data_train)
    x_test, y_test = baseline_processing(data_test)
    y_test['Transported'] = model.predict(x_test)
    y_test.to_csv('predictions/output_1_baseline.csv', index=False)


# get_baseline_predictions(train, test)
