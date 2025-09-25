import sys
import os
import pandas
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, load_object
from src.utils import save_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predicted(self, features_df):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features_df)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info(f'Error in Prediction {e}')
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 date:float,
                 bedrooms:float,
                 bathrooms:float,
                 sqft_living:float,
                 sqft_lot:float,
                 floors:float,
                 waterfront:float,
                 view:float,
                 condition:float,
                 grade:float,
                 sqft_above:float,
                 sqft_basement:float,
                 yr_built:float,
                 yr_renovated:float,
                 zipcode:float):
        self.date = date
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.sqft_living = sqft_living
        self.sqft_lot = sqft_lot
        self.floors = floors
        self.waterfront = waterfront
        self.view = view
        self.condition = condition
        self.grade = grade
        self.sqft_above = sqft_above
        self.sqft_basement = sqft_basement
        self.yr_built = yr_built
        self.yr_renovated = yr_renovated
        self.zipcode = zipcode

    def get_dataframe(self):
        try:
            custom_data_input_dict = {
                'date':[self.date],
                'bedrooms':[self.bedrooms],
                'bathrooms':[self.bathrooms],
                'sqft_living' : [self.sqft_living],
                'sqft_lot' : [self.sqft_lot],
                'floors' : [self.floors],
                'waterfront' : [self.waterfront],
                'view' : [self.view],
                'condition' : [self.condition],
                'grade' : [self.grade],
                'sqft_above' : [self.sqft_above],
                'sqft_basement' : [self.sqft_basement],
                'yr_built' : [self.yr_built],
                'yr_renovated' : [self.yr_renovated],
                'zipcode':[self.zipcode]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f'DataFrame Created {df}')
            return df
        except Exception as e:
            logging.info(f'Error in Creating DataFrame {e}')
            raise CustomException(e,sys)

if __name__ == "__main__":
    # Example input
    custom_data = CustomData(
        date=20141013.0,   # Example YYYYMMDD
        bedrooms=3,
        bathrooms=2,
        sqft_living=1800,
        sqft_lot=5000,
        floors=1.0,
        waterfront=0,
        view=0,
        condition=3,
        grade=7,
        sqft_above=1600,
        sqft_basement=200,
        yr_built=1995,
        yr_renovated=0,
        zipcode=98178
    )

    # Convert to DataFrame
    input_df = custom_data.get_dataframe()

    # Prediction
    pipeline = PredictionPipeline()
    prediction = pipeline.predicted(input_df)

    print("✅ Predicted House Price:", prediction)
