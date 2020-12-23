### customer_transactions_history

[**Documentaion to this project solution**](doc.md)

### Aligning on Expectations

The purpose of this test is to assess your problem solving skills (see https://i.datachef.co/problem-solvers), creativity and experience in tackling a data science question. 

You don't need to finish the assignment 100% or perfect your code before posting it. Partial and incomplete solutions will be reviewed as well.

### Feel free to impress your future teammates with your skills in:

- Python 3 or Scala
- Visualizations in Matplolib or Bokeh
- Data processing in Pandas, SQL, etc
- Data structures and algorithms
- Git (yes, we will read all your commit messages!)

### Your code won't be evaluated for

- Extensive/fancy documentation or diagrams. A readable README.md along with clear in-line comments is enough.

It is completely up to you which data processing framework, machine learning library, or visualization tool, etc you use. 
Just provide us explanation why you have chosen that specific tool for this specific task and in the context of this assignment.

### Non Functional Requirements

1. If using Python, feel free to use Jupyter notebooks as a dev environment but also make sure to include production ready code as .py files along with the model serialization artifacts. 
2. Create a separate file for loading the saved model and evaluating the results locally. Include clear in-line instructions. 
3. Code must be forked and pushed on your personal Github (temporarily).

### Problem statement

The attached dataset contains three features about the customers of a business.
- Customer ID
- Product
- Time stamp

Our goal is to build a model to facilitate the analysis and prediction of the customer behavior and product dynamics. You're free to pick any architecture and any data model that you deem suitable. Make sure to explain your rationale for picking the model. We're interested in models that deliver reasonably high accuracy and are ideally flexible and powerful enough for adding more features. Here are a few basic questions:

- Create an ordered (descending) plot that shows the total number of transactions per customer from the most active customer to the least active one. 
- Given any product ID, create a plot to show its transaction frequency per month for the year 2018. 
- Build a model to predict the total number of transactions for the next three months per customer anywhere in 2019. For example, given all data up to the end of January 2019, predict the size of the transactions between Feb 1st and April 30th for each customer. Then, measure the performance of your model with appropriate metrics and visuals. 
- At any time, what are the top 5 products that drove the highest sales over the last six months? Do you see a seasonality effect in this data set?

Feel free to state and solve any other questions that you find interesting about this data set. Attention to good coding practices and style will be noted. Also, feel free to implement more than one model and compare their performance.

### Final Words

This mini-assignment is designed carefully to stretch your skills beyond everyday data science tasks. The reason is: we, in the first place, are looking for great software craftsmen. Being at DataChef, you need to be ready for challenges beyond boring data plumbing tasks and finding your way in understanding (sometimes vague) requirements by best guessing (and documenting) your scenarios.

We value your time and effort and unlike many other companies, we take this test seriously and will put quality time to check your code with an open mind, objective and no-bigoted review process.

