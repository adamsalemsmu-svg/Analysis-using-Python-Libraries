# Analysis using Python Libraries

This project demonstrates an end-to-end data analysis and machine-learning workflow using popular Python libraries: **Pandas**, **NumPy**, and **scikit-learn**. We load the classic Iris dataset, perform data cleaning and preprocessing, train a logistic regression classifier, and evaluate its performance. The project also illustrates how to integrate SQL-style queries and natural-language queries via a chatbot interface (CSQLbot).

## Dataset

We use the Iris dataset from the `sklearn.datasets` module. The dataset contains 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and a target label indicating the species.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- Optional: pandasql (to run SQL queries on a DataFrame)
- Optional: A natural-language SQL chatbot (e.g., CSQLbot) to translate natural language into SQL and execute queries.

Install dependencies with:

```bash
pip install pandas numpy scikit-learn pandasql
```

## Script Overview

The main script, **analysis.py**, contains the following functions:

- `load_iris_df()`: Loads the Iris dataset into a pandas DataFrame and cleans feature names.
- `train_logistic_regression(df)`: Splits the data into training and test sets, scales features, and trains a logistic regression classifier.
- `run_sql_query(df, query)`: Executes a SQL query on a DataFrame using pandasql.
- `natural_language_to_sql(question)`: Maps a simple natural-language question to a SQL query (example implementation).
- `run_natural_language_query(df, question)`: Converts a natural-language question to SQL and runs it against the DataFrame.

At the end of the script, it trains the model and prints the classification report, followed by running an example natural-language query.

## Usage

1. Clone the repository and install dependencies.
2. Run the analysis script:

```bash
python analysis.py
```

The script will print training accuracy and a classification report, then execute a sample natural-language query such as "average sepal length for each species" and display the results.

## Example SQL Query

To get the average sepal length by species via SQL:

```sql
SELECT species, AVG("sepal length (cm)") AS avg_sepal_length
FROM df
GROUP BY species;
```

## Future Improvements

- Expand the `natural_language_to_sql` function to handle more complex questions.
- Integrate a CSQLbot or LangChain agent to automatically translate natural language to SQL using LLMs.
- Include additional visualizations and interactive dashboards using libraries like matplotlib or plotly.
