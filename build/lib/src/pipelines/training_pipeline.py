from src.logger import logging

from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    Data_model = DataIngestion()
    train_arr,test_arr = Data_model.initiate_data_ingestion()
    logging.info(f'Train Set {train_arr} test_set {test_arr}')
    print(f'{train_arr,test_arr}')

    Data_trans = DataTransformation()
    train_arr,test_arr,_ = Data_trans.initiate_data_transformation(train_arr,test_arr)
    print(f'Transformation is Done')
    logging.info('Transformation is Done')

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr,test_arr)
