# Spaceship Titanic Kaggle Competition

## Overview
Welcome to the Spaceship Titanic Kaggle competition showcase! This repository encapsulates my journey through the competition, culminating in a top 5% performance with a precision score of 0.8057 for binary classification. Initially undertaken as a learning project, it now stands as a valuable addition to my portfolio, showcasing the skills and techniques I've acquired in the realm of machine learning.

## Project Structure

### `baseline_model.py`
Here lies the initial simple model, serving as a baseline for subsequent comparisons.

### `baseline_pipeline.py`
The first iteration of the pipeline, presenting a straightforward approach to data processing and modeling.

### `explanatory.py`
A visual treat! This file offers informative graphs and visualizations, shedding light on the feature engineering process.

### `feature_union_pipe.py`
The star of the show! This file hosts the main pipeline, featuring the final prediction model along with an extensive grid search.

### Feature Engineering
Witness the transformation! Engage with features like 'Travelling_Solo', 'GroupSize', 'Cabin_Deck', 'Cabin_Number', and more, crafted to enhance the predictive power of the model.

### `GeneralCleaner` Class
A bespoke transformer class, meticulously designed for cleaning and imputing various features.

### `FeatureUnionCustom` Class
A custom feature union class orchestrating the collaboration of different transformers.

### `NumericTransformer` Class
A nimble transformer specializing in the numerical feature transformation domain.

### Pipeline Components
1. **Data Cleaning and Imputation:** The `GeneralCleaner` class takes charge of cleaning and imputing, ensuring a pristine dataset.
2. **Feature Selection:** The `FeatureSelector` class elegantly separates numeric and categorical features.
3. **Imputation:** Enter the `CustomImputer` class, stepping in to fill in missing values with precision.
4. **One-Hot Encoding:** The `CustomDummify` class executes a tailored one-hot encoding strategy, with an option to drop the first column.
5. **Scaling:** The `CustomScaler` class handles the numerical features, offering a choice between standard and robust scaling.

### Model Selection
Embark on a journey through various models, including Random Forest, Gradient Boosting, AdaBoost, SVM, and more. The final contenders for optimization are LightGBM, Random Forest, and XGBoost.

### Model Optimization
The thrilling grid search unfolds! Tune hyperparameters for selected models, adjusting settings like the number of estimators, learning rate, and maximum depth for optimal performance.

### `fit_model` and `save_predictions_to_csv` Functions
These functions encapsulate the essence of model fitting and the seamless creation of prediction CSV files.

### `preview_df` Function
Peek behind the curtain! The `preview_df` function unveils a transformed DataFrame, providing a snapshot of the preprocessing magic.

## Usage
Embark on your own exploration:
1. Review the Python files, particularly `feature_union_pipe.py`.
2. Tailor configurations and hyperparameters to your requirements.
3. Execute the code to immerse yourself in the training and evaluation of models.
4. Delve into additional visualizations and analyses in `explanatory.py`.
5. Utilize the provided functions to fit the final model and save predictions.

## Acknowledgments
This project owes its existence to the Spaceship Titanic Kaggle competition. A heartfelt thank you to Kaggle for the dataset and the community for fostering valuable insights and discussions. Feel free to contribute and enhance this project—it's an open canvas for collaborative improvement.
