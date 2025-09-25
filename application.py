from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        try:
            # ✅ Required fields for House Price Prediction
            required_fields = [
                'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'
            ]

            # Validation
            for field in required_fields:
                if not request.form.get(field):
                    return render_template('form.html', error=f'Please fill in the {field} field.')

            # ✅ Create CustomData object with house input
            data = CustomData(
                date=float(request.form.get('date')),
                bedrooms=float(request.form.get('bedrooms')),
                bathrooms=float(request.form.get('bathrooms')),
                sqft_living=float(request.form.get('sqft_living')),
                sqft_lot=float(request.form.get('sqft_lot')),
                floors=float(request.form.get('floors')),
                waterfront=float(request.form.get('waterfront')),
                view=float(request.form.get('view')),
                condition=float(request.form.get('condition')),
                grade=float(request.form.get('grade')),
                sqft_above=float(request.form.get('sqft_above')),
                sqft_basement=float(request.form.get('sqft_basement')),
                yr_built=float(request.form.get('yr_built')),
                yr_renovated=float(request.form.get('yr_renovated')),
                zipcode=float(request.form.get('zipcode'))
            )

            # Convert input into DataFrame
            final_new_data = data.get_dataframe()

            # Run prediction
            predict_pipeline = PredictionPipeline()
            pred = predict_pipeline.predicted(final_new_data)

            results = round(pred[0], 2)

            return render_template('form.html', final_result=results)

        except ValueError:
            return render_template('form.html', error='Please enter valid numeric values for all fields.')
        except Exception as e:
            return render_template('form.html', error=f'An error occurred: {str(e)}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
