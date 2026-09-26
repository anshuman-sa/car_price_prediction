import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# ==========================
# Load Dataset
# ==========================

car = pd.read_csv("quikr_car.csv")

print("Original Dataset")
print(car.head())

# ==========================
# Data Cleaning
# ==========================

# Keep only numeric year
car = car[car["year"].str.isnumeric()]
car["year"] = car["year"].astype(int)

# Remove "Ask For Price"
car = car[car["Price"] != "Ask For Price"]

# Convert price to integer
car["Price"] = car["Price"].str.replace(",", "", regex=False)
car["Price"] = car["Price"].astype(int)

# Clean kms_driven
car["kms_driven"] = car["kms_driven"].str.replace(" kms", "", regex=False)
car["kms_driven"] = car["kms_driven"].str.replace(",", "", regex=False)

car = car[car["kms_driven"].str.isnumeric()]
car["kms_driven"] = car["kms_driven"].astype(int)

# Remove missing fuel type
car = car.dropna()

print("\nCleaned Dataset")
print(car.head())

# ==========================
# Features and Target
# ==========================

X = car.drop(columns="Price")
y = car["Price"]

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Encode Categorical Columns
# ==========================

categorical = ["name", "company", "fuel_type"]

transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="passthrough"
)

# ==========================
# Model
# ==========================

model = Pipeline([
    ("preprocessor", transformer),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# ==========================
# Accuracy
# ==========================

prediction = model.predict(X_test)

print("\nR2 Score :", r2_score(y_test, prediction))

# ==========================
# Save Model
# ==========================

joblib.dump(model, "car_price_model.pkl")

print("\nModel Saved Successfully!")

# ==========================
# User Prediction
# ==========================

print("\nEnter Car Details")

name = input("Car Name : ")
company = input("Company : ")
year = int(input("Year : "))
kms = int(input("Kilometers Driven : "))
fuel = input("Fuel Type : ")

input_df = pd.DataFrame({
    "name": [name],
    "company": [company],
    "year": [year],
    "kms_driven": [kms],
    "fuel_type": [fuel]
})

price = model.predict(input_df)

print("\nEstimated Car Price: ₹{:,.0f}".format(price[0]))