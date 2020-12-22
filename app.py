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

@app.command('pr', help='Predict the total number of transactions for the next three months per customer anywhere in 2019')
def prediction(customer_id: Optional[str]=None, month: Optional[int]=1, train: bool=False):
    if train == False and customer_id == None:
        raise typer.BadParameter('prediction mode needs customer id.')
    
    
    if train:
        train_model.train(selected_month=month)
    else:
        predict_model.predict(customer_id)


@app.command()
def top_product():
    pass

if __name__ == '__main__':
    app()