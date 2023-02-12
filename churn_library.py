# library doc string
'''
This file is used for predicting customer churn

Author: Nguyen Huy Anh

Date: 19th Jan
'''

# import libraries
import os
import joblib
import numpy as np
import shap
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # read dataframe with pandas
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    img_folder = "./images/eda"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # print shape of dataframe
    print(df.shape)

    # print number of null values of each column
    print(df.isnull().sum())

    # print the statistic metrics of each column
    print(df.describe())

    # create Churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # plot histograms
    hist_cols = ['Churn', 'Customer_Age']
    for col in hist_cols:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.hist(df[col])
            ax.set_title(f'{col} Histogram')
            plt.tight_layout()
            fig.savefig(f"{img_folder}/{col}_distribution.png")

    # plot marital status value counts with normalizing
    if 'Marital_Status' in df.columns:
        if df['Marital_Status'].dtype == 'object':
            fig, ax = plt.subplots(figsize=(20, 10))
            df['Marital_Status'].value_counts(
                normalize=True).plot(
                kind='bar', ax=ax)
            ax.set_title('Marital Status Value Counts after normalizing')
            plt.tight_layout()
            fig.savefig(f"{img_folder}/marital_status_distribution.png")

    # distribution plot of total_trans_ct
    if 'Total_Trans_Ct' in df.columns:
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.distplot(df['Total_Trans_Ct'], ax=ax)
        ax.set_title('Total Trans Ct')
        plt.tight_layout()
        fig.savefig(f"{img_folder}/total_transaction_distribution.png")

    # correlation heatmap of dataframe
    corr = df.corr()
    if corr.size > 1:
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2, ax=ax)
        ax.set_title('Dataframe correlation')
        plt.tight_layout()
        fig.savefig(f"{img_folder}/heatmap.png")
    else:
        print("Not enough columns for correlation heatmap")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming\
                 variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        # create a new list to store the encoded values
        encoded_lst = []
        # group the dataframe by the current category and calculate the mean of
        # the response variable
        category_groups = df.groupby(category).mean()[response]
        # iterate through each value in the current category column
        for val in df[category]:
            # append the mean response value for the current category
            encoded_lst.append(category_groups.loc[val])
        # create a new column in the dataframe with the encoded values
        df[f'{category}_Churn'] = encoded_lst

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming \
                variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # turn each categorical column into a new column with propotion of churn
    # for each category
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = encoder_helper(df, cat_columns, response)

    y = df[response]

    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('./images/results/rf_results.png')

    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # set up the plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # plot the SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.legend(["SHAP"])

    # plot the feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]
    plt.bar(range(X_data.shape[1]), importances[indices], alpha=0.5)
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.legend(["Importance"])

    # save the plot
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # roc curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store feature importances plot
    feature_importance_plot(cv_rfc.best_estimator_, X_train,
                            './images/results/feature_importances.png')

    # store model results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    train_models(X_train, X_test, y_train, y_test)
