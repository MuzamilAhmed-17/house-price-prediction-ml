# Project: House Price Prediction (King County Dataset)
Machine Learning regression project to predict house prices using feature engineering, log transformation, and Linear Regression (R² = 0.65).

1️⃣ Import Libraries

import pandas as pd

import numpy as np

→ pandas → Data loading, cleaning, feature manipulation

→ numpy → Numerical operations, log transformation

2️⃣ Load Dataset

df = pd.read_csv('house_price_prediction.csv')

df.shape

df.head()

→ Load King County housing dataset

→ Initial exploration for understanding data shape and preview

3️⃣ Drop Unnecessary Columns

df = df.drop(["statezip", "country", "street"], axis=1)

→ These columns are not useful for price prediction

→ Reduce noise in the model

4️⃣ Encode Categorical Feature: City

from sklearn.preprocessing import OneHotEncoder

city_col = df['city']

city_reshaped = city_col.values.reshape(-1,1)

ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid dummy variable trap

city_encoded = ohe.fit_transform(city_reshaped)

city_df = pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out(['city']))

df = df.drop('city', axis=1)

df = pd.concat([df, city_df], axis=1)

→ City is categorical → One-Hot Encoding applied

→ Prevents linear regression from misinterpreting city as ordinal

drop='first' avoids multicollinearity

5️⃣ Date Processing & Feature Engineering

df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")

df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

df['day'] = df['date'].dt.day

df["age"] = df["year"] - df["yr_built"]

df['renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)

→ Split date into year, month, day

→ Calculate age of house → important for price prediction

→ Convert renovation info to binary → 0 not renovated, 1 renovated

6️⃣ Drop Redundant Columns

df = df.drop(["date", "yr_built", "sqft_above", "sqft_basement", "sqft_lot", "yr_renovated", "bedrooms","year"], axis=1)

→ Drop columns causing multicollinearity

→ Keep only features that improve model stability

7️⃣ Remove Zero Prices and Log Transform Target

df = df[df['price'] > 0]

df["price"] = np.log(df['price'])

→ Remove rows with zero price → prevent log(0) errors

→ Log-transform price → reduce skew, improve linear regression fit

8️⃣ Split Data & Train Model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

X = df.drop('price', axis=1)

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

→ Train-test split 80-20

→ Train Linear Regression model

→ Predict on test set

9️⃣ Evaluate Model

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error = ", mse)

print("R2 = ", r2)

→ MSE (Mean Squared Error) → measure of error magnitude

→ R² Score → how much variance explained by model (~0.65 after log-transform)


