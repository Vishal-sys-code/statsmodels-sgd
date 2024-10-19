# CPS ASEC Analysis using SGD-based OLS with Gradient Clipping

This notebook demonstrates how to use our statsmodels-like SGD implementation with gradient clipping to analyze CPS ASEC microdata.

## Setup and Data Download

First, let's set up our environment and download the necessary data:

```python
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import statsmodels_sgd.api as sm_sgd

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

# Create dummy variables
df = pd.get_dummies(df, columns=['educ_category', 'A_SEX'], drop_first=True)

print("Data preparation complete.")
```

## SGD-based OLS Regression Analysis

Now we'll use our custom SGD-based OLS implementation:

```python
# Prepare the data for our model
X = df[['A_AGE', 'A_SEX_2', 'educ_category_HS grad',
        'educ_category_Some college', 'educ_category_Bachelor\'s or higher']]
y = df['PEARNVAL']

# Add constant term
X = sm.add_constant(X)

# Initialize and fit our model
# Note: Adjust hyperparameters as needed
model = sm_sgd.OLS(
    n_features=X.shape[1], learning_rate=0.01, epochs=1000, batch_size=1000, clip_value=1.0
)
model.fit(X, y)

# Print the summary
print(model.summary())

# Calculate R-squared
y_pred = model.predict(X)
r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print(f"\nR-squared: {r_squared:.4f}")
```

## Interpretation

The output of this regression model tells us:

1. How earnings change with each year of age, all else being equal.
2. The earnings difference between males and females, controlling for age and education.
3. How different levels of education are associated with earnings, compared to having less than a high school education.
4. The overall fit of the model (R-squared).

Note that this SGD-based implementation may produce slightly different results compared to traditional OLS due to its iterative nature and the use of gradient clipping. The advantage is that it can handle larger datasets more efficiently and provides some level of differential privacy through gradient clipping.

## Comparison with Traditional OLS

To see how our SGD-based implementation compares with traditional OLS, let's run the same analysis using statsmodels:

```python
import statsmodels.api as sm

# Fit traditional OLS model
ols_model = sm.OLS(y, X).fit()

# Print summary
print(ols_model.summary())
```

Compare the coefficients, standard errors, and R-squared values between the two methods. The SGD-based method should provide similar results, but with the added benefits of scalability and privacy preservation.

## Cleanup

Finally, let's clean up our downloaded file:

```python
# Remove the downloaded CSV file
os.remove("pppub24.csv")
print("Cleanup complete.")
```

This notebook provides a complete, reproducible example of downloading CPS ASEC microdata and performing a regression analysis using our custom SGD-based OLS implementation with gradient clipping. It demonstrates how to use this method with real-world data and compares it to traditional OLS.
