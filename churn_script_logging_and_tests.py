import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_for_testing = import_data(PTH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    except ValueError as err:
        logging.error(
            "Testing import_data: The file name must be of type string")

    try:
        assert df_for_testing.shape[0] > 0
        assert df_for_testing.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        assert os.path.exists("./images/eda")
        assert os.path.exists("./images/eda/Churn_distribution.png")
        assert os.path.exists("./images/eda/Customer_Age_distribution.png")
        assert os.path.exists("./images/eda/marital_status_distribution.png")
        assert os.path.exists(
            "./images/eda/total_transaction_distribution.png")
        assert os.path.exists("./images/eda/heatmap.png")
        logging.info("Testing perfom_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_data: Folder images/eda doesn't exists or the folder doesn't\
			 contain enough images")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df_for_testing = encoder_helper(df, cat_columns, response)
        logging.info("Testing encode_helper: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Tesing encode_helper: The first parameter must be type of dataframe")
        raise err
    except KeyError as err:
        logging.error("Testing encode_helper: The column '{}' does not exist in the dataframe"
                      .format(response))
        raise err

    try:
        for cols in cat_columns:
            assert cols in df.columns
    except AssertionError as err:
        logging.error(
            "Testing encode_helper: The column '{}' does not exist in the dataframe" .format(cols))
        raise err

    try:
        for cols in cat_columns:
            new_cols = cols + "_" + "Churn"
            assert new_cols in df_for_testing.columns
    except AssertionError as err:
        logging.error(
            "Testing encode_helper: The column '{}' was not created".format(new_cols))
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train_for_testing, X_test_for_testing,\
            y_train_for_testing, y_test_for_testing = perform_feature_engineering(df, response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Tesing perform_feature_engineering: The first parameter must be type of \
			dataframe")
        raise err
    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering: The column '{}' does not exist in the \
			dataframe" .format(response))
        raise err

    try:
        assert X_train_for_testing.shape[0] > 0
        assert X_test_for_testing.shape[0] > 0
        assert X_train_for_testing.shape[0] == len(y_train_for_testing)
        assert X_test_for_testing.shape[0] == len(y_test_for_testing)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: This function doesn't split the dataframe")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: No dataset to train")
        raise err

    try:
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./images/results")
        assert os.path.exists("./images/results/feature_importances.png")
        assert os.path.exists("./images/results/logistic_results.png")
        assert os.path.exists("./images/results/rf_results.png")
        assert os.path.exists("./images/results/roc_curve_result.png")
    except AssertionError as err:
        logging.error(
            "Testing train_models: This function doesn't train models")
        raise err


if __name__ == "__main__":
    PTH = './data/bank_data.csv'
    test_import(cls.import_data)
    df = cls.import_data(PTH)
    test_eda(cls.perform_eda)
    cls.perform_eda(df)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    response = "Churn"
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, response)
    test_train_models(cls.train_models)
