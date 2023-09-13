import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.shape)
# print(test.shape)
#
# print(train.isna().sum())
# print(test.isna().sum())
#
# print(train.nunique())
# print(train.dtypes)
#
# plt.figure(figsize=(6, 6))
# train['Transported'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True,
#                                              textprops={'fontsize': 16}).set_title("Target distribution")
# plt.show()
#
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
# print(train[qual_feats].head())

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


# FURTHER EXPLANATORY TESTING:


df = train.copy()

df[['Group', 'Member']] = df['PassengerId'].str.split('_', expand=True)
df[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)

group_count = df.groupby("Group")["Member"].count()
gc = df['Group'].value_counts().sort_index()
# print(group_count.equals(gc))

df['Travelling_Solo'] = df['Group'].apply(lambda x: x not in set(gc[gc > 1].index))
df['Group_Size'] = df.groupby('Group')['Member'].transform('count')
df['Cabin_Number'].fillna(df['Cabin_Number'].median(), inplace=True)
df['Cabin_Number'] = df['Cabin_Number'].astype(int)


def plot_groups():
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.countplot(x="Group_Size", hue="Transported", data=df, palette="Set2")
    plt.title("Group Size vs Transported")

    plt.subplot(1, 2, 2)
    sns.countplot(x="Travelling_Solo", hue="Transported", data=df, palette="Set2")
    plt.title("Travelling Solo vs Transported")

    plt.tight_layout()
    plt.show()


# plot_groups()


def plot_cabin():
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Cabin_Deck', hue='Transported', palette='Set2',
                  order=["A", "B", "C", "D", "E", "F", "G", "T"])
    plt.title("Cabin Deck Distribution")

    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='Cabin_Side', hue='Transported', palette='Set2')
    plt.title("Cabin Side Distribution")
    # plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.histplot(data=df, x='Cabin_Number', hue='Transported', palette='Set2')
    plt.title('Cabin Number Distribution')
    plt.xticks(list(range(0, 1900, 300)))
    for position in [300, 600, 900, 1200, 1500]:
        plt.vlines(position, ymin=0, ymax=550, color="black")

    plt.show()


# plot_cabin()

cbins = [0, 300, 600, 900, 1200, 1500, float('inf')]
clabels = ['Cabin_Region1', 'Cabin_Region2', 'Cabin_Region3', 'Cabin_Region4', 'Cabin_Region5', 'Cabin_Region6']

df['Cabin_Region'] = pd.cut(df['Cabin_Number'], bins=cbins, labels=clabels, right=False)


def plot_cabin_region():
    plt.figure(figsize=(20, 25))
    sns.countplot(x="Cabin_Region", hue="Transported", data=df, palette="Set2")
    plt.title("Cabin_Region Distribution")
    plt.show()


# plot_cabin_region()

df["Total_Expenditure"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
df["No_Spending"] = (df["Total_Expenditure"] == 0)


def plot_total_exp():
    plt.figure(figsize=(15, 6))
    sns.histplot(data=df, x='Total_Expenditure', hue='Transported', palette="Set2", bins=200)
    plt.ylim(0, 400)
    plt.xlim(0, 10000)
    plt.title('Total Expenditure Distribution')
    plt.show()


# plot_total_exp()

exp_mean = round(df["Total_Expenditure"].mean(), 2)
exp_median = df["Total_Expenditure"].median()
# print(exp_mean, exp_median)

df['Expenditure_Category'] = pd.cut(df['Total_Expenditure'],
                                    bins=[-1, 0, exp_median, exp_mean, float('inf')],
                                    labels=['No_Expense', 'Low_Expense', 'Medium_Expense', 'High_Expense'])


def plot_expenditure_cats():
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='No_Spending', hue='Transported', palette='Set2')
    plt.title('No Spending Distribution')
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='Expenditure_Category', hue='Transported', palette='Set2')
    plt.title('No Spending Distribution')
    plt.show()


# plot_expenditure_cats()

print(df.head(20))
