import typer
from typing import Optional
from typer.main import param_path_convertor
from models import predict_model, train_model
from visualization import visuals


app = typer.Typer()

@app.command(help='Create an ordered (descending) plot that shows the total number of transactions per customer from the most active customer to the least active one.')
def customer_transactions():
    visuals.plot_sales()

@app.command(help='Given any product ID, create a plot to show its transaction frequency per month for the year 2018.')
def product_transactions(product_id: str):
    visuals.plot_sale_frequency_by_mouth(product_id)

@app.command(help='Train the model for prediction.')
def train(epochs: int=50):
    train_model.train(epochs=epochs)

@app.command( help='Predict the total number of transactions for the next three months per customer anywhere in 2019')
def predict(customer_id: int, month: int):
    predict_model.predict(customer_id, month)

@app.command(help='The top 5 products that drove the highest sales over the last six months')
def top_product(year: int, month: int):
    visuals.plot_top_5(year, month)

if __name__ == '__main__':
    app()