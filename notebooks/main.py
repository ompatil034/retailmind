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



###DAY 2

# Imports the three Day 2 classification models.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Creates the Decision Tree, Random Forest and SVM models.
day2_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Trains each model using the same Day 1 preprocessing and compares the results.
results = {}

for name, classifier in day2_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
        "Pipeline": pipeline
    }

# Prints the performance of all three models.
for name, result in results.items():
    print(name)
    print("Accuracy:", result["Accuracy"])
    print("Precision:", result["Precision"])
    print("Recall:", result["Recall"])
    print("ROC-AUC:", result["ROC-AUC"])
    print()

# Selects the model with the highest churn recall.
best_model_name = max(results, key=lambda name: results[name]["Recall"])
best_model = results[best_model_name]["Pipeline"]

print("Best Model:", best_model_name)

# Saves the Day 2 champion model alongside the Day 1 model.
joblib.dump(best_model, "models/day2_champion.joblib")



###DAY 3

# Imports the Day 3 tools for tuning and cross-validation.
from sklearn.model_selection import GridSearchCV, cross_val_score

# Creates simple parameter grids for the three Day 2 models.
parameter_grids = {
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 10],
        "classifier__min_samples_split": [2, 5]
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None]
    },
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"]
    }
}

# Tunes each model using GridSearchCV with cross-validation.
tuned_results = {}

for name, classifier in day2_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    grid_search = GridSearchCV(
        pipeline,
        parameter_grids[name],
        cv=5,
        scoring="recall"
    )

    grid_search.fit(X_train, y_train)

    tuned_results[name] = {
        "Model": grid_search.best_estimator_,
        "Best Parameters": grid_search.best_params_,
        "CV Recall": grid_search.best_score_
    }

# Prints the best parameters and cross-validation recall for each model.
for name, result in tuned_results.items():
    print(name)
    print("Best Parameters:", result["Best Parameters"])
    print("CV Recall:", result["CV Recall"])
    print()

# Compares training and validation scores to check bias and variance.
for name, result in tuned_results.items():

    tuned_model = result["Model"]

    train_score = tuned_model.score(X_train, y_train)

    validation_scores = cross_val_score(
        tuned_model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(name)
    print("Training Accuracy:", train_score)
    print("Validation Accuracy:", validation_scores.mean())
    print()

# Selects the tuned model with the highest cross-validation recall.
best_tuned_model_name = max(
    tuned_results,
    key=lambda name: tuned_results[name]["CV Recall"]
)

best_tuned_model = tuned_results[best_tuned_model_name]["Model"]

print("Day 3 Champion:", best_tuned_model_name)

# Saves the final tuned Day 3 champion model.
joblib.dump(best_tuned_model, "models/churn_v3.joblib")



###DAY 4

# Imports the Day 4 tools for feature selection, PCA and ensemble models.
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

# Creates simple new features from the bank churn dataset.
X_day4 = X.copy()
X_day4["Balance_Age_Ratio"] = X_day4["Balance"] / X_day4["Age"]
X_day4["Salary_Age_Ratio"] = X_day4["EstimatedSalary"] / X_day4["Age"]
X_day4["Balance_Per_Product"] = X_day4["Balance"] / X_day4["NumOfProducts"]

# Splits the Day 4 data into training and testing data.
X_train_day4, X_test_day4, y_train_day4, y_test_day4 = train_test_split(
    X_day4, y, test_size=0.2, random_state=42, stratify=y
)

# Separates the numerical and categorical columns after feature engineering.
numeric_day4 = X_day4.select_dtypes(include=["int64", "float64"]).columns
categorical_day4 = X_day4.select_dtypes(include=["str"]).columns

# Creates preprocessing for the new Day 4 features.
preprocessor_day4 = ColumnTransformer([
    ("num", StandardScaler(), numeric_day4),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_day4)
])

# Creates the Random Forest and XGBoost ensemble models.
ensemble_models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

# Trains and compares the Day 4 ensemble models.
day4_results = {}

for name, classifier in ensemble_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor_day4),
        ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train_day4, y_train_day4)

    y_pred = pipeline.predict(X_test_day4)
    y_prob = pipeline.predict_proba(X_test_day4)[:, 1]

    day4_results[name] = {
        "Model": pipeline,
        "Accuracy": accuracy_score(y_test_day4, y_pred),
        "Precision": precision_score(y_test_day4, y_pred),
        "Recall": recall_score(y_test_day4, y_pred),
        "ROC-AUC": roc_auc_score(y_test_day4, y_prob)
    }

# Prints the performance of the Day 4 ensemble models.
for name, result in day4_results.items():
    print(name)
    print("Accuracy:", result["Accuracy"])
    print("Precision:", result["Precision"])
    print("Recall:", result["Recall"])
    print("ROC-AUC:", result["ROC-AUC"])
    print()

# Selects the Day 4 model with the highest churn recall.
day4_champion_name = max(
    day4_results,
    key=lambda name: day4_results[name]["Recall"]
)

day4_champion = day4_results[day4_champion_name]["Model"]

print("Day 4 Champion:", day4_champion_name)

# Saves the final Day 4 churn model.
joblib.dump(day4_champion, "models/churn_v4.joblib")


###DAY 5

# Imports the Day 5 tools needed to build and train the ANN.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Preprocesses the same bank churn data for the neural network.
X_train_ann = preprocessor_day4.fit_transform(X_train_day4)
X_test_ann = preprocessor_day4.transform(X_test_day4)

# Creates a simple feed-forward ANN with a sigmoid output.
ann = Sequential([
    Dense(16, activation="relu", input_shape=(X_train_ann.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compiles the ANN for binary churn prediction.
ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Stops training when validation loss stops improving.
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Trains the ANN using validation data and early stopping.
ann.fit(
    X_train_ann,
    y_train_day4,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Makes churn predictions using the trained ANN.
ann_prob = ann.predict(X_test_ann, verbose=0).ravel()
ann_pred = (ann_prob >= 0.5).astype(int)

# Calculates and prints the ANN performance metrics.
ann_accuracy = accuracy_score(y_test_day4, ann_pred)
ann_precision = precision_score(y_test_day4, ann_pred)
ann_recall = recall_score(y_test_day4, ann_pred)
ann_roc_auc = roc_auc_score(y_test_day4, ann_prob)

print("ANN Accuracy:", ann_accuracy)
print("ANN Precision:", ann_precision)
print("ANN Recall:", ann_recall)
print("ANN ROC-AUC:", ann_roc_auc)

# Compares the ANN recall with the Day 4 classical champion.
print("Day 4 Champion Recall:", day4_results[day4_champion_name]["Recall"])
print("ANN Recall:", ann_recall)

# Saves the trained Day 5 ANN model.
ann.save("models/churn_ann.keras")
