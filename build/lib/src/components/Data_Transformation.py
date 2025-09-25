# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Feature Engineering Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self_):
        logging.info('Working started Data Transformation')
        try:
            numerical_col = ['date', 'bedrooms', 'bathrooms',
                             'sqft_living','sqft_lot', 'floors',
                             'waterfront', 'view', 'condition', 'grade',
                              'sqft_above', 'sqft_basement', 'yr_built',
                             'yr_renovated', 'zipcode']

            logging.info('Pipeline is initiated')
            num_trans = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_trans, numerical_col)
            ])
            logging.info('preprocessor is created')
            return preprocessor

        except Exception as e:
            logging.info(f'Error in Get Transformation {e}')
            CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n{test_df.head().to_string()}')

            preprocessor_obj = self.get_data_transformation_obj()

            target_col_name = 'price'
            drop_cols = [target_col_name,'ID']
            # Splitting features and target
            input_feature_train_df = train_df.drop(columns=drop_cols, axis=1, errors='ignore')
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=drop_cols, axis=1, errors='ignore')
            target_feature_test_df = test_df[target_col_name]

            # Apply the transformation
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Tranformation Problem here")
            logging.info("Applied preprocessing object on training and testing data")

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('combined features')
            # Save the preprocessor object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj= preprocessor_obj
            )
            logging.info('Preprocessor pickle is created at: %s', self.data_transformation_config.preprocessor_obj_file)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e:
            logging.info(f'Error in Transformation {e}')
            CustomException(e,sys)
