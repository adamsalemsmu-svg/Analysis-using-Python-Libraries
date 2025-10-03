# Analysis using Python Libraries

This project demonstrates a small data analysis and machine‑learning workflow using popular Python libraries: **Pandas**, **NumPy** and **scikit‑learn**. We load the classic Iris dataset, perform simple preprocessing, train a classifier, and evaluate its performance. A future extension includes integrating a natural‑language SQL bot (CSQLbot) to query the data interactively.

## Dataset

We use the Iris dataset from the `sklearn.datasets` module. The dataset contains 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and a target label indicating the species.

## Requirements

- Python 3.7 +
- pandas
- numpy
- scikit‑learn
- *Optional:* `pandasql` (to run SQL queries on a DataFrame)
- *Optional:* An LLM‑powered SQL chatbot (e.g., CSQLbot) to translate natural language into SQL

You can install the required libraries with:

```bash
pip install pandas numpy scikit-learn pandasql
```

## Usage

Run the analysis script:

```bash
python analysis.py
```

This will:

1. Load the Iris dataset into a pandas DataFrame.
2. Split the data into training and test sets.
3. Scale features using `StandardScaler`.
4. Train a logistic regression classifier.
5. Output accuracy and a classification report.

### Example SQL Queries

If you install `pandasql`, you can run SQL queries on the DataFrame. For example:

```python
from pandasql import sqldf
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

query = '''
SELECT *
FROM df
WHERE "sepal length (cm)" > 5.0
  AND "petal width (cm)" < 1.5
LIMIT 10;
'''

result = sqldf(query, {'df': df})
print(result)
```

To integrate with a natural‑language SQL bot like **CSQLbot**, provide prompts such as “Show me the first 5 iris flowers with petal length greater than 4 cm” and use the generated SQL with `sqldf` to query the DataFrame.

## License

This project is licensed under the MIT License.
