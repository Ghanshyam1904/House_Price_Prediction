import os
import pandas as pd
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Working is started')
        try:
            df = pd.read_csv(os.path.join('notebook/Dataset','House_Data.csv'))
            logging.info('DataSet Reading Successfully')

            os.makedirs(os.path.dirname(self.data_ingestion.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False,header=True)

            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)
            logging.info('Training and Testing Data is Done')

            train_set.to_csv(self.data_ingestion.train_data_path,header=True,index=False)
            test_set.to_csv(self.data_ingestion.test_data_path,header=True,index=False)

            logging.info('Train Csv is done')
            return (
              self.data_ingestion.train_data_path,
              self.data_ingestion.test_data_path
            )
        except Exception as e:
            logging.info(f'Error in Data Ingestion {e}')
            CustomException(e,sys)