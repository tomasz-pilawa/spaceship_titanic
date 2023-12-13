import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

pd.set_option('display.width', None)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

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

# UNCOMMENT BELOW FOR PLOTS (plot_cabin) & LINE184 TO WORK, COMMENT FOR IMPUTATION TO WORK PROPERLY
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

# LINE184
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


def visualise_num_transformed():
    cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Expenditure']

    for col in cols:
        df[col] = np.log1p(df[col])

    x = 1
    plt.figure(figsize=(15, 25))
    for i in cols:
        plt.subplot(6, 2, x)
        sns.histplot(data=df, x=i, color="green")
        plt.ylim(0, 0.2)
        plt.title(f"{i} Distribution")
        x += 1
    plt.show()


# visualise_num_transformed()

df['Amenities_Used'] = df[['RoomService', 'Spa', 'VRDeck', 'FoodCourt', 'ShoppingMall']].gt(0).sum(axis=1)
df['Spending_Service'] = df[['RoomService', 'Spa', 'VRDeck']].sum(axis=1)
df['Spending_Shopping'] = df[['FoodCourt', 'ShoppingMall']].sum(axis=1)
df['Surname'] = df['Name'].str.split().str[-1]
df['Family_Size'] = df.groupby('Surname')['Surname'].transform('count')


def plot_families():
    plt.figure(figsize=(16, 9))
    sns.countplot(data=df, x='Family_Size', hue='Transported', palette="Set2")
    plt.title('Family Size Distribution')
    plt.show()


# plot_families()

# ADVANCED IMPUTATION ###############################

# print(pd.crosstab(df['Group'], df['HomePlanet']))

HP_bef = df['HomePlanet'].isna().sum()

df['HomePlanet'] = df.groupby('Group')['HomePlanet'].transform(
    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else np.nan)

# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', df['HomePlanet'].isna().sum())


def plot_cabin_deck_missing():
    crosstab = pd.crosstab(df['Cabin_Deck'], df['HomePlanet'])
    crosstab['Missing'] = df.groupby('Cabin_Deck')['HomePlanet'].apply(lambda x: x.isna().sum())
    print(crosstab)
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, fmt='d')
    plt.title('Cross-Tabulation: Group vs HomePlanet')
    plt.xlabel('HomePlanet')
    plt.ylabel('Group')
    plt.show()


# plot_cabin_deck_missing()
HP_bef = df['HomePlanet'].isna().sum()

cabin_to_homeplanet = {'A': 'Europa', 'B': 'Europa', 'C': 'Europa', 'T': 'Europa', 'G': 'Earth'}
df['HomePlanet'] = df['Cabin_Deck'].map(cabin_to_homeplanet).fillna(df['HomePlanet'])

HP_after = df['HomePlanet'].isna().sum()

# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', HP_after)


# print(pd.crosstab(df['Surname'], df['HomePlanet']))

HP_bef = df['HomePlanet'].isna().sum()

df['HomePlanet'] = df.groupby('Surname')['HomePlanet'].transform(
    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty and not pd.isna(x.name) else x).fillna(df['HomePlanet'])

# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', df['HomePlanet'].isna().sum())


def impute_home_planet(row):
    # this one is a little bit arbitraty from the net - not sure if much better than simple imputation
    if pd.isnull(row['HomePlanet']):
        if row['Cabin_Deck'] == 'D':
            return 'Mars'
        else:
            return 'Earth'
    else:
        return row['HomePlanet']


# df['HomePlanet'] = df.apply(impute_home_planet, axis=1)

# this is simpler, since earth is most common anyways, so with most frequent it will get filled
# print(df.groupby('HomePlanet')['HomePlanet'].count())
df.loc[(df['Cabin_Deck'] == 'D') & df['HomePlanet'].isna(), 'HomePlanet'] = 'Mars'

# print('#HomePlanet missing values after:', df['HomePlanet'].isna().sum())


def plot_surname_group():
    crosstab = pd.crosstab(df['Group'], df['Surname'], margins=True)
    print(crosstab)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=crosstab['All'].index, data=crosstab['All'])
    plt.ylabel('Count')
    plt.title('Number of Unique Surname by Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# plot_surname_group()

HP_bef = df['Surname'].isna().sum()

df['Surname'] = df.groupby('Group')['Surname'].transform(
    lambda t: t.fillna(t.mode()[0]) if not t.mode().empty else 'Unknown')

print('#Surname missing values before:', HP_bef)
print('#Surname missing values after:', df['Surname'].isna().sum())

for n in ['Cabin_Deck', 'Cabin_Side']:
    HP_bef = df[n].isna().sum()

    df[n] = df.groupby('Group')[n].transform(
        lambda t: t.fillna(t.mode()[0]) if not t.mode().empty else np.nan)
    df[n] = df.groupby(['HomePlanet', 'Destination', 'Travelling_Solo'])[n].transform(
        lambda x: x.fillna(x.mode()[0]) if not x.mode().empty and not pd.isna(x.name) else x).fillna(df[n])

    print(f'#{n} missing values before:', HP_bef)
    print(f'#{n} missing values after:', df[n].isna().sum())

CN_bef = df['Cabin_Number'].isna().sum()

# for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
#     missing_values_count = df[(df['Cabin_Deck'] == deck) & (df['Cabin_Number'].isna())].shape[0]
#     print(f'Deck {deck}: Missing Values Count: {missing_values_count}')

for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    X_CN = df.loc[~(df['Cabin_Number'].isna()) & (df['Cabin_Deck'] == deck), 'Group']
    y_CN = df.loc[~(df['Cabin_Number'].isna()) & (df['Cabin_Deck'] == deck), 'Cabin_Number']
    X_test_CN = df.loc[(df['Cabin_Number'].isna()) & (df['Cabin_Deck'] == deck), 'Group']

    # print(deck)
    # print(X_CN.shape, y_CN.shape, X_test_CN.shape)

    model_CN = LinearRegression()
    model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
    preds_CN = model_CN.predict(X_test_CN.values.reshape(-1, 1))
    df.loc[(df['Cabin_Number'].isna()) & (df['Cabin_Deck'] == deck), 'Cabin_Number'] = preds_CN.astype(int)

print('#Cabin_number missing values before:', CN_bef)
print('#Cabin_number missing values after:', df['Cabin_Number'].isna().sum())

A_bef = df['Age'].isna().sum().sum()

df.loc[df['Age'].isna(), 'Age'] = \
    df.groupby(['HomePlanet', 'No_Spending', 'Travelling_Solo', 'Cabin_Deck'])['Age'].transform(
        lambda x: x.fillna(x.median()))[df.loc[df['Age'].isna(), 'Age'].index]

print('#Age missing values before:', A_bef)
print('#Age missing values after:', df['Age'].isna().sum())


def plot_cryo_no_spending():
    crosstab = pd.crosstab(df['CryoSleep'], df['No_Spending'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, fmt='d')
    plt.title('Cross-Tabulation: CryoSleep vs No_Spending')
    plt.xlabel('CryoSleep')
    plt.ylabel('No_Spending')
    plt.show()


# plot_cryo_no_spending()

HP_bef = df['CryoSleep'].isna().sum()

df['CryoSleep'] = df.groupby('No_Spending')['CryoSleep'].transform(lambda cs: cs.fillna(cs.mode()[0]))

print('#CryoSleep missing values before:', HP_bef)
print('#CryoSleep missing values after:', df['CryoSleep'].isna().sum())

# print(df.isna().sum())
# # print(df.head(20))
