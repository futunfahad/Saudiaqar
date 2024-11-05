import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv("penguins_size.csv")

# Drop rows with any NaN values
df = df.dropna()

# Separate features and target variable
x = df.drop("species", axis=1)  # All columns except species
y = df["species"]  # Target variable

# Define categorical and numerical features
categorical_features = ["island", "sex"]
numerical_features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)

# Create a pipeline that first preprocesses the data and then fits the model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Fit the model
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, "penguin_model_with_island.joblib")
