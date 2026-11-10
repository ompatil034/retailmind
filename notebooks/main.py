###DAY 1


import warnings
warnings.filterwarnings("ignore")
# Imports the libraries needed for the churn model.
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Loads the bank customer churn dataset.
df = pd.read_csv("data/churn.csv")

# Shows the dataset structure, missing values and target balance.
print(df.info())
print(df.isnull().sum())
print(df["Exited"].value_counts())

# Removes unnecessary columns and separates inputs from the target.
X = df.drop(columns=["Exited", "CustomerId", "Surname"])
y = df["Exited"]

# Splits the dataset into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Separates numerical and categorical columns.
numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["str"]).columns

# Scales numerical columns and converts categorical columns into numbers.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

# Creates and trains the baseline Logistic Regression pipeline.
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Calculates and prints the required model performance metrics.
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Saves the complete pipeline so it can be used in later days.
joblib.dump(model, "models/churn_pipeline.joblib")

# Uses the saved pipeline to predict whether a new customer will exit.
def predict(customer_data):
    saved_model = joblib.load("models/churn_pipeline.joblib")
    return saved_model.predict(customer_data)


