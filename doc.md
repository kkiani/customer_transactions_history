### docs for customer_transactions_history
this package contains both production and development files for predicting and visulizing customer_transactions_history dataset.

### project structure
- all dataset including raw data and processed data is in data directory
- jupyter notebook file has been used for developmnet tests and are accessible through notebook folders
- the `app.py` file is the root file of this package
- svr results were lower than lstm model for that reason the lstm model have been choosed for production, but svr is also availabe through `prediction-development` notebook.
- model serialization artifacts including both trained model and scalers are available inside the `models/serialized` path.

### installing dependencies
to install dependencies using pip, use `requierments-dev.txt` for development and `requierments-pro.txt` for production enviorment.
**Attionsion:** this package has been developed and tested under python 3. python 2 is not recommended.

```shell
pip install -r requierments-pro.txt
```

### how to run?
this package provides shell interface with rich help. to get the help:
```shell
python app.py --help
```

| command        | description           | arguments  |
| -------------- |:-------------:| ----------:|
| customer-transactions     | Create an ordered (descending) plot that shows the total number of transactions per customer from the most active customer to the least active one. | - |
| product-transactions       | Predict the total number of transactions for the next three months per customer anywhere in 2019     | product_id(int) |
| top-product  | the top 5 products that drove the highest sales over the last six months   |  year(int), month(int) |
| predict | Predict the total number of transactions for the next three months per customer anywhere in 2019 | customer_id(int), month(int) |
| train | Train the model for prediction. | epochs(int) defualt is 50 |

example:
```shell
python app.py predict 1000178 1
```
