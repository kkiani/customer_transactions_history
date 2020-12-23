import typer
from typing import Optional
from typer.main import param_path_convertor
from models import predict_model, train_model


app = typer.Typer()

@app.command('ct')
def customer_transactions():
    typer.echo('ok')

@app.command('pt', help='Products Transaction frequency per month for the year 2018')
def product_transactions(product_id: int):
    typer.echo('f')

@app.command('train', help='Train the model for prediction.')
def train(month: int, epochs: int=50):
    typer.secho(f'🍻 You are training the model with month={month}, make sure to pass same month in prediction', fg=typer.colors.MAGENTA)
    train_model.train(selected_month=month, epochs=epochs)

@app.command('predict', help='Predict the total number of transactions for the next three months per customer anywhere in 2019')
def predict(customer_id: int, month: int):
    predict_model.predict(customer_id, month)

@app.command()
def top_product():
    pass

if __name__ == '__main__':
    app()