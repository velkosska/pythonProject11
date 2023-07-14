import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing()

df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

df['target'] = housing.target

print(df.head())
print(df.info())

df.dropna(inplace=True)

statistics = df.describe()
print(statistics)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print('R^2 score:', score)

new_data = [[-122.25, 37.85, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]]
prediction = model.predict(new_data)
print('Prediction:', prediction)