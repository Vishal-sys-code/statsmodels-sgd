# CPS ASEC Multiple Regression Analysis

This notebook demonstrates how to download the CPS ASEC microdata and perform a multiple regression analysis using the person-level file.

## Setup and Data Download

First, let's set up our environment and download the necessary data:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
import zipfile
import io
import os

# URL for the CPS ASEC microdata
url = "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip"

# Download the zip file
print("Downloading CPS ASEC microdata...")
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the person-level file
print("Extracting person-level data...")
z.extract("pppub24.csv")

print("Data download and extraction complete.")
```

## Load and Prepare Data

Now, let's load the data and prepare it for analysis:

```python
# Load the person-level data
print("Loading person-level data...")
df = pd.read_csv('pppub24.csv')

# Select variables for analysis
# A_AGE: Age
# A_HGA: Educational attainment
# A_SEX: Sex
# PEARNVAL: Total person's earnings

# Filter for adults aged 18+ with positive earnings
df = df[(df['A_AGE'] >= 18) & (df['PEARNVAL'] > 0)]

# Create dummy variables for education (A_HGA)
# We'll group education into 4 categories:
df['educ_category'] = pd.cut(df['A_HGA'],
                             bins=[0, 38, 39, 42, 100],
                             labels=['Less than HS', 'HS grad', 'Some college', 'Bachelor\'s or higher'])

print("Data preparation complete.")
```

## Regression Analysis

Now we'll set up and run our regression model:

```python
# Set up the regression model
X = df[['A_AGE', 'A_SEX', 'educ_category']]
y = df['PEARNVAL']

# Create a pipeline with preprocessing and regression
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['A_AGE']),
        ('cat', OneHotEncoder(drop='first', sparse=False), ['A_SEX', 'educ_category'])
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
print("Fitting regression model...")
model.fit(X, y)

# Print the coefficients
feature_names = ['Age', 'Sex_Female'] + [f'Educ_{cat}' for cat in ['HS grad', 'Some college', 'Bachelor\'s or higher']]
coefficients = model.named_steps['regressor'].coef_

print("\nRegression Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f}")

# Print the intercept
print(f"\nIntercept: {model.named_steps['regressor'].intercept_:.2f}")

# Calculate R-squared
r_squared = model.score(X, y)
print(f"\nR-squared: {r_squared:.4f}")
```

## Interpretation

The output of this regression model tells us:

1. How earnings change with each year of age, all else being equal.
2. The earnings difference between males and females, controlling for age and education.
3. How different levels of education are associated with earnings, compared to having less than a high school education.
4. The overall fit of the model (R-squared).

Remember that this is a simplified model and doesn't account for many other factors that could influence earnings. It's meant as a demonstration of how to work with CPS ASEC data rather than a comprehensive analysis of earnings determinants.

## Cleanup

Finally, let's clean up our downloaded file:

```python
# Remove the downloaded CSV file
os.remove("pppub24.csv")
print("Cleanup complete.")
```

This notebook provides a complete, reproducible example of downloading CPS ASEC microdata and performing a multiple regression analysis. It can be easily incorporated into a Jupyter Book or CI pipeline as part of a larger package.
