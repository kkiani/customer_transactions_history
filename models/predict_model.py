import pandas as pd
import datetime
import typer
import joblib
from tensorflow.keras.models import load_model

def predict(customer_id: int, selected_month: int):
    
    # loading dataset
    df = pd.read_csv('data/trans_per_month.csv', index_col='customer_id')

    # loading scaler models
    x_scaler = joblib.load('models/serialized/x_scaler.mod')
    y_scaler = joblib.load('models/serialized/y_scaler.mod')

    # selecting data
    prediction_start_date = str(datetime.date(2019, selected_month, 1))
    customer_history = df.loc[customer_id, df.columns.to_series().between(str(datetime.date(2017, selected_month, 1)), prediction_start_date)].values  

    customer_history = x_scaler.transform(customer_history.reshape(1, -1))  
    customer_history = customer_history.reshape(1, customer_history.shape[1], 1)

    model = load_model('models/serialized/lstm_model')

    prediction = model.predict(customer_history)
    prediction = y_scaler.inverse_transform(prediction)[:, 0]

    typer.secho(f'💡 prediction for customer with id {customer_id} is {prediction[0]:.2f} number of transactions', fg=typer.colors.BRIGHT_YELLOW)
