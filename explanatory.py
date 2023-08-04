import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
print(test.shape)

print(train.isna().sum())
print(test.isna().sum())

# print(train.nunique())
# print(train.dtypes)

# plt.figure(figsize=(6, 6))
# train['Transported'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True,
#                                              textprops={'fontsize': 16}).set_title("Target distribution")
# plt.show()

# plt.figure(figsize=(10, 4))
# sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
# plt.title('Age distribution')
# plt.xlabel('Age (years)')
# plt.show()

# Create a new feature that indicates whether the passenger is a child, adolescent or adult.

exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def make_histograms_numerical():
    fig = plt.figure(figsize=(16, 28))
    for i, var_name in enumerate(exp_feats):
        # Left plot
        ax = fig.add_subplot(5, 2, 2 * i + 1)
        sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')

        # Right plot (truncated)
        ax = fig.add_subplot(5, 2, 2 * i + 2)
        sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
        plt.ylim([0, 100])
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def make_histograms_numerical_2():
    fig, axes = plt.subplots(nrows=len(exp_feats), ncols=2, figsize=(12, 4 * len(exp_feats)))
    for i, var_name in enumerate(exp_feats):
        # Left plot
        ax = axes[i, 0]
        sns.histplot(data=train, x=var_name, ax=ax, bins=30, kde=False, hue='Transported', common_norm=False)
        # Right plot (truncated)
        ax = axes[i, 1]
        sns.histplot(data=train, x=var_name, ax=ax, bins=30, kde=True, hue='Transported', common_norm=False)
        ax.set_ylim([0, 100])  # Truncate the y-axis for both subplots
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# make_histograms_numerical()
# make_histograms_numerical_2()


# Create a new feature that tracks the total expenditure across all 5 amenities.
# Create a binary feature to indicate if the person has not spent anything. (i.e. total expenditure is 0).
# Take the log transform to reduce skew.


cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']


def plot_cat_feats():
    fig = plt.figure(figsize=(10, 16))
    for i, var_name in enumerate(cat_feats):
        ax = fig.add_subplot(4, 1, i + 1)
        sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')
        # ax.set_title(var_name)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05, hspace=0.4)
    plt.show()


# plot_cat_feats()

# We might consider dropping the VIP column to prevent overfitting.

qual_feats = ['PassengerId', 'Cabin', 'Name']
print(train[qual_feats].head())

# We can extract the group and group size from the PassengerId feature.
# We can extract the deck, number and side from the cabin feature.
# We could extract the surname from the name feature to identify families.

# FEATURE ENGINEERING

age_bins = [0, 12, 18, 25, 30, 50, float('inf')]
age_labels = ['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+']
train['Age_group'] = pd.cut(train['Age'], bins=age_bins, labels=age_labels, right=False)


def plot_age_bins():
    plt.figure(figsize=(10, 4))
    g = sns.countplot(data=train, x='Age_group', hue='Transported',
                      order=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])
    plt.title('Age group distribution')
    plt.show()


# plot_age_bins()

