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
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

# Creates and trains the baseline Logistic Regression pipeline.
model = Pipeline(
    [("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
)

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
    "SVM": SVC(probability=True, random_state=42),
}

# Trains each model using the same Day 1 preprocessing and compares the results.
results = {}

for name, classifier in day2_models.items():

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

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
        "Pipeline": pipeline,
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
        "classifier__min_samples_split": [2, 5],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
    },
    "SVM": {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]},
}

# Tunes each model using GridSearchCV with cross-validation.
tuned_results = {}

for name, classifier in day2_models.items():

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    grid_search = GridSearchCV(pipeline, parameter_grids[name], cv=5, scoring="recall")

    grid_search.fit(X_train, y_train)

    tuned_results[name] = {
        "Model": grid_search.best_estimator_,
        "Best Parameters": grid_search.best_params_,
        "CV Recall": grid_search.best_score_,
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
        tuned_model, X_train, y_train, cv=5, scoring="accuracy"
    )

    print(name)
    print("Training Accuracy:", train_score)
    print("Validation Accuracy:", validation_scores.mean())
    print()

# Selects the tuned model with the highest cross-validation recall.
best_tuned_model_name = max(
    tuned_results, key=lambda name: tuned_results[name]["CV Recall"]
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
preprocessor_day4 = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_day4),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_day4),
    ]
)

# Creates the Random Forest and XGBoost ensemble models.
ensemble_models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
}

# Trains and compares the Day 4 ensemble models.
day4_results = {}

for name, classifier in ensemble_models.items():

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor_day4),
            ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train_day4, y_train_day4)

    y_pred = pipeline.predict(X_test_day4)
    y_prob = pipeline.predict_proba(X_test_day4)[:, 1]

    day4_results[name] = {
        "Model": pipeline,
        "Accuracy": accuracy_score(y_test_day4, y_pred),
        "Precision": precision_score(y_test_day4, y_pred),
        "Recall": recall_score(y_test_day4, y_pred),
        "ROC-AUC": roc_auc_score(y_test_day4, y_prob),
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
day4_champion_name = max(day4_results, key=lambda name: day4_results[name]["Recall"])

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
ann = Sequential(
    [
        Dense(16, activation="relu", input_shape=(X_train_ann.shape[1],)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compiles the ANN for binary churn prediction.
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Stops training when validation loss stops improving.
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Trains the ANN using validation data and early stopping.
ann.fit(
    X_train_ann,
    y_train_day4,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
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


### DAY 6

# Imports the Day 6 TensorFlow and Keras tools needed for the CNN.
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout

# Loads the Fashion-MNIST training and testing images.
(X_train_cnn, y_train_cnn), (X_test_cnn, y_test_cnn) = fashion_mnist.load_data()

# Normalises the image pixel values between 0 and 1.
X_train_cnn = X_train_cnn / 255.0
X_test_cnn = X_test_cnn / 255.0

# Adds one channel dimension so the images can be given to the CNN.
X_train_cnn = X_train_cnn[..., None]
X_test_cnn = X_test_cnn[..., None]

# Creates a simple CNN for classifying the 10 product categories.
cnn = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# Compiles the CNN for product image classification.
cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Trains the CNN on the Fashion-MNIST images.
cnn.fit(
    X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_split=0.2, verbose=1
)

# Evaluates and prints the CNN accuracy on the test images.
cnn_loss, cnn_accuracy = cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)

print("CNN Accuracy:", cnn_accuracy)

# Creates and prints the confusion matrix for the CNN predictions.
cnn_predictions = cnn.predict(X_test_cnn, verbose=0)
cnn_predicted_classes = cnn_predictions.argmax(axis=1)

from sklearn.metrics import confusion_matrix

cnn_confusion_matrix = confusion_matrix(y_test_cnn, cnn_predicted_classes)

print("Confusion Matrix:")
print(cnn_confusion_matrix)

# Saves the Day 6 product image classifier.
cnn.save("models/catalog_cnn.keras")


###DAY 7

# Imports the Day 7 tools for transfer learning and object detection.
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from ultralytics import YOLO

# Creates a pretrained MobileNetV2 model without its original output layer.
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

# Freezes the pretrained layers for transfer learning.
base_model.trainable = False

# Adds a new classification head for the 10 product categories.
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(10, activation="softmax")(x)

transfer_model = Model(inputs=base_model.input, outputs=output)

# Compiles the transfer learning model.
transfer_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Resizes Fashion-MNIST images and converts them to three channels.
X_train_transfer = tf.image.resize(X_train_cnn, (96, 96))
X_test_transfer = tf.image.resize(X_test_cnn, (96, 96))

X_train_transfer = tf.image.grayscale_to_rgb(X_train_transfer)
X_test_transfer = tf.image.grayscale_to_rgb(X_test_transfer)

# Trains the new classification head while keeping the base model frozen.
transfer_model.fit(
    X_train_transfer,
    y_train_cnn,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# Fine-tunes the top layers of the pretrained model.
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

transfer_model.fit(
    X_train_transfer,
    y_train_cnn,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# Evaluates transfer learning accuracy against the Day 6 CNN.
transfer_loss, transfer_accuracy = transfer_model.evaluate(
    X_test_transfer, y_test_cnn, verbose=0
)

print("Day 6 CNN Accuracy:", cnn_accuracy)
print("Transfer Learning Accuracy:", transfer_accuracy)

# Saves the Day 7 transfer learning classifier.
transfer_model.save("models/product_transfer_model.keras")

# Loads a pretrained YOLO model for product object detection.
yolo_model = YOLO("yolov8n.pt")

# Detects and locates objects in a catalog image using bounding boxes.
yolo_results = yolo_model("product_image.jpg")

# Saves the detected image with its bounding boxes.
for result in yolo_results:
    result.save(filename="detected_product.jpg")


###DAY 8


# Builds segmentation, background removal and Grad-CAM explainability for the vision module.
import cv2
import numpy as np
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# Creates simple product masks from the Fashion-MNIST images.
X_segmentation = X_train_cnn.astype("float32")
y_segmentation = (X_segmentation > 0.1).astype("float32")

# Builds a small U-Net-style segmentation model.
segmentation_input = Input(shape=(28, 28, 1))

conv1 = Conv2D(16, 3, activation="relu", padding="same")(segmentation_input)
pool1 = MaxPooling2D(2)(conv1)

conv2 = Conv2D(32, 3, activation="relu", padding="same")(pool1)
pool2 = MaxPooling2D(2)(conv2)

bridge = Conv2D(64, 3, activation="relu", padding="same")(pool2)

up1 = UpSampling2D(2)(bridge)
up1 = Concatenate()([up1, conv2])
conv3 = Conv2D(32, 3, activation="relu", padding="same")(up1)

up2 = UpSampling2D(2)(conv3)
up2 = Concatenate()([up2, conv1])
conv4 = Conv2D(16, 3, activation="relu", padding="same")(up2)

segmentation_output = Conv2D(1, 1, activation="sigmoid")(conv4)

segmentation_model = Model(segmentation_input, segmentation_output)

# Compiles and trains the segmentation model.
segmentation_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

segmentation_model.fit(
    X_segmentation,
    y_segmentation,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# Predicts a product mask and places the product on a white background.
sample_image = X_test_cnn[0]
predicted_mask = segmentation_model.predict(sample_image[None, ...], verbose=0)[0]

binary_mask = (predicted_mask > 0.5).astype("uint8")

product_image = (sample_image * 255).astype("uint8")
white_background = np.full_like(product_image, 255)

clean_product_image = cv2.bitwise_and(
    product_image, product_image, mask=binary_mask[:, :, 0]
)

inverse_mask = cv2.bitwise_not(binary_mask[:, :, 0])

clean_background = cv2.bitwise_and(
    white_background, white_background, mask=inverse_mask
)

catalog_image = cv2.add(clean_product_image, clean_background)

cv2.imwrite("clean_catalog_image.png", catalog_image)

# Creates a Grad-CAM heatmap for the transfer learning classifier.
last_conv_layer = base_model.get_layer("Conv_1")

grad_model = Model(
    transfer_model.input, [last_conv_layer.output, transfer_model.output]
)

gradcam_image = X_test_transfer[0:1]

with tf.GradientTape() as tape:
    convolution_output, predictions = grad_model(gradcam_image)
    predicted_class = tf.argmax(predictions[0])
    class_score = predictions[:, predicted_class]

gradients = tape.gradient(class_score, convolution_output)
weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

heatmap = tf.reduce_sum(convolution_output[0] * weights, axis=-1)

heatmap = tf.maximum(heatmap, 0)
heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
heatmap = cv2.resize(heatmap.numpy(), (96, 96))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

original_image = np.uint8(X_test_transfer[0].numpy() * 255)
gradcam_result = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcam_product.png", gradcam_result)

# Saves the Day 8 segmentation model.
segmentation_model.save("models/product_segmentation.keras")


###DAY 9
# Loads the sales data and creates a daily time series for one store and product.
sales_df = pd.read_csv("data/train.csv", parse_dates=["date"])

sales_series = (
    sales_df[
        (sales_df["store"] == sales_df["store"].iloc[0])
        & (sales_df["item"] == sales_df["item"].iloc[0])
    ]
    .groupby("date")["sales"]
    .sum()
    .sort_index()
    .asfreq("D")
    .fillna(0)
)

# Keeps the most recent 20 percent of the sales data for testing.
split_index = int(len(sales_series) * 0.8)

train_series = sales_series.iloc[:split_index]
test_series = sales_series.iloc[split_index:]

# Scales the sales data using only the training data.
train_min = train_series.min()
train_max = train_series.max()

scaled_series = (sales_series - train_min) / (train_max - train_min)

# Creates look-back windows for next-day demand prediction.
look_back = 30
X_lstm = []
y_lstm = []

for i in range(look_back, len(scaled_series)):
    X_lstm.append(scaled_series.iloc[i - look_back : i].values)
    y_lstm.append(scaled_series.iloc[i])

X_lstm = tf.convert_to_tensor(X_lstm, dtype=tf.float32)
y_lstm = tf.convert_to_tensor(y_lstm, dtype=tf.float32)

X_lstm = tf.expand_dims(X_lstm, axis=2)

# Splits the windows into training and testing data without shuffling.
window_split = split_index - look_back

X_train_lstm = X_lstm[:window_split]
X_test_lstm = X_lstm[window_split:]
y_train_lstm = y_lstm[:window_split]
y_test_lstm = y_lstm[window_split:]

# Builds a simple LSTM model for next-day demand forecasting.
lstm_model = Sequential(
    [tf.keras.layers.Input(shape=(look_back, 1)), tf.keras.layers.LSTM(50), Dense(1)]
)

# Compiles the LSTM model.
lstm_model.compile(optimizer="adam", loss="mse")

# Trains the LSTM model using validation data.
lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    verbose=1,
)

# Converts predictions and actual values back to their original sales units.
scaled_predictions = lstm_model.predict(X_test_lstm, verbose=0).ravel()

actual_sales = y_test_lstm.numpy() * (train_max - train_min) + train_min
predicted_sales = scaled_predictions * (train_max - train_min) + train_min

# Calculates and prints MAE and RMSE in the original sales units.
mae = tf.reduce_mean(tf.abs(actual_sales - predicted_sales))
rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_sales - predicted_sales)))

print("MAE:", float(mae))
print("RMSE:", float(rmse))

# Plots the actual and predicted demand values.
forecast_results = pd.DataFrame({"Actual": actual_sales, "Predicted": predicted_sales})

forecast_results.plot(title="Actual vs Predicted Demand", figsize=(10, 5))

# Saves the Day 9 demand forecasting model.
lstm_model.save("models/demand_lstm.keras")


###DAY 10

# Creates multi-step windows for seven-day demand forecasting.
forecast_horizon = 7

X_seq2seq = []
y_seq2seq = []
target_indexes = []

for i in range(look_back, len(scaled_series) - forecast_horizon + 1):
    X_seq2seq.append(scaled_series.iloc[i - look_back : i].values)
    y_seq2seq.append(scaled_series.iloc[i : i + forecast_horizon].values)
    target_indexes.append(i)

X_seq2seq = np.array(X_seq2seq, dtype="float32")[..., None]
y_seq2seq = np.array(y_seq2seq, dtype="float32")[..., None]
target_indexes = np.array(target_indexes)

# Splits the sequence data while keeping the latest period for testing.
train_mask = target_indexes + forecast_horizon <= split_index
test_mask = target_indexes >= split_index

X_train_seq2seq = X_seq2seq[train_mask]
y_train_seq2seq = y_seq2seq[train_mask]

X_test_seq2seq = X_seq2seq[test_mask]
y_test_seq2seq = y_seq2seq[test_mask]

# Builds the encoder for reading the historical demand window.
encoder_input = tf.keras.layers.Input(shape=(look_back, 1), name="encoder_input")

encoder_output, encoder_state_h, encoder_state_c = tf.keras.layers.LSTM(
    64, return_sequences=True, return_state=True, name="encoder"
)(encoder_input)

# Builds the decoder for producing seven future demand values.
decoder_input = tf.keras.layers.RepeatVector(forecast_horizon)(encoder_state_h)

decoder_output = tf.keras.layers.LSTM(64, return_sequences=True, name="decoder")(
    decoder_input, initial_state=[encoder_state_h, encoder_state_c]
)

# Adds attention so the decoder can focus on useful historical days.
attention_output, attention_scores = tf.keras.layers.AdditiveAttention(
    name="attention"
)([decoder_output, encoder_output], return_attention_scores=True)

combined_output = tf.keras.layers.Concatenate()([decoder_output, attention_output])

forecast_output = tf.keras.layers.TimeDistributed(Dense(1), name="forecast")(
    combined_output
)

# Creates and compiles the Seq2Seq attention model.
seq2seq_model = Model(inputs=encoder_input, outputs=forecast_output)

seq2seq_model.compile(optimizer="adam", loss="mse")

# Trains the model to forecast the next seven days.
seq2seq_model.fit(
    X_train_seq2seq,
    y_train_seq2seq,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    verbose=1,
)

# Predicts seven future demand values for every test window.
scaled_seq2seq_predictions = seq2seq_model.predict(X_test_seq2seq, verbose=0)

actual_seq2seq = y_test_seq2seq.squeeze(-1) * (train_max - train_min) + train_min

predicted_seq2seq = (
    scaled_seq2seq_predictions.squeeze(-1) * (train_max - train_min) + train_min
)

# Calculates the multi-step MAE and RMSE in original sales units.
seq2seq_mae = tf.reduce_mean(tf.abs(actual_seq2seq - predicted_seq2seq))

seq2seq_rmse = tf.sqrt(tf.reduce_mean(tf.square(actual_seq2seq - predicted_seq2seq)))

print("Day 9 LSTM RMSE:", float(rmse))
print("Seq2Seq MAE:", float(seq2seq_mae))
print("Seq2Seq RMSE:", float(seq2seq_rmse))

# Plots the actual and predicted demand for the first test forecast.
multi_step_results = pd.DataFrame(
    {"Actual": actual_seq2seq[0], "Predicted": predicted_seq2seq[0]}
)

multi_step_results.plot(title="Seven-Day Actual vs Predicted Demand", figsize=(10, 5))

# Creates a model that returns the attention weights.
attention_model = Model(inputs=encoder_input, outputs=attention_scores)

attention_weights = attention_model.predict(X_test_seq2seq[:1], verbose=0)[0]

# Plots the attention given to each historical day.
attention_dataframe = pd.DataFrame(
    attention_weights,
    columns=[f"History Day {day + 1}" for day in range(look_back)],
    index=[f"Forecast Day {day + 1}" for day in range(forecast_horizon)],
)

attention_dataframe.T.plot(title="Attention Weights", figsize=(10, 5))

# Saves the Day 10 multi-step demand forecasting model.
seq2seq_model.save("models/demand_seq2seq_attention.keras")

###DAY 11

# Finds the product class with the lowest Day 6 classification accuracy.
class_accuracy_before = []

for class_number in range(10):
    class_mask = y_test_cnn == class_number
    class_accuracy = np.mean(
        cnn_predicted_classes[class_mask] == y_test_cnn[class_mask]
    )
    class_accuracy_before.append(class_accuracy)

rare_class = int(np.argmin(class_accuracy_before))

# Selects images from the lowest-performing product class.
rare_images = X_train_cnn[y_train_cnn == rare_class]
rare_images = rare_images * 2.0 - 1.0

# Builds a generator for creating synthetic product images.
generator = Sequential(
    [
        tf.keras.layers.Input(shape=(100,)),
        Dense(7 * 7 * 128),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(
            64, 4, strides=2, padding="same", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            1, 4, strides=2, padding="same", activation="tanh"
        ),
    ]
)

# Builds a discriminator for identifying real and synthetic images.
discriminator = Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        Conv2D(64, 4, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Conv2D(128, 4, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation="sigmoid"),
    ]
)

# Creates the optimizers and loss function for GAN training.
generator_optimizer = tf.keras.optimizers.Adam(0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002)
gan_loss = tf.keras.losses.BinaryCrossentropy()

# Trains the generator and discriminator on the selected product class.
batch_size = 64
epochs = 20

rare_dataset = (
    tf.data.Dataset.from_tensor_slices(rare_images)
    .shuffle(len(rare_images))
    .batch(batch_size)
)

for epoch in range(epochs):
    for real_images in rare_dataset:
        current_batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal((current_batch_size, 100))

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(real_images, training=True)
            generated_output = discriminator(generated_images, training=True)

            generator_loss = gan_loss(tf.ones_like(generated_output), generated_output)

            discriminator_loss = gan_loss(
                tf.ones_like(real_output), real_output
            ) + gan_loss(tf.zeros_like(generated_output), generated_output)

        generator_gradients = generator_tape.gradient(
            generator_loss, generator.trainable_variables
        )

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )

        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

# Generates synthetic images for the lowest-performing product class.
synthetic_images = generator.predict(tf.random.normal((1000, 100)), verbose=0)

synthetic_images = (synthetic_images + 1.0) / 2.0
synthetic_images = np.clip(synthetic_images, 0.0, 1.0)

# Keeps synthetic images with enough visible detail.
image_variance = np.var(synthetic_images, axis=(1, 2, 3))
realistic_mask = image_variance > 0.01
synthetic_images = synthetic_images[realistic_mask]
synthetic_labels = np.full(len(synthetic_images), rare_class)

# Adds the synthetic images to the original classifier training data.
X_train_augmented = np.concatenate([X_train_cnn, synthetic_images], axis=0)

y_train_augmented = np.concatenate([y_train_cnn, synthetic_labels], axis=0)

# Fine-tunes a copy of the Day 6 classifier using the augmented data.
augmented_cnn = tf.keras.models.clone_model(cnn)
augmented_cnn.set_weights(cnn.get_weights())

augmented_cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

augmented_cnn.fit(
    X_train_augmented,
    y_train_augmented,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# Measures per-class accuracy after adding the synthetic images.
augmented_predictions = augmented_cnn.predict(X_test_cnn, verbose=0).argmax(axis=1)

class_accuracy_after = []

for class_number in range(10):
    class_mask = y_test_cnn == class_number
    class_accuracy = np.mean(
        augmented_predictions[class_mask] == y_test_cnn[class_mask]
    )
    class_accuracy_after.append(class_accuracy)

print("Augmented Class:", rare_class)
print("Accuracy Before:", class_accuracy_before[rare_class])
print("Accuracy After:", class_accuracy_after[rare_class])

# Saves the Day 11 generator and augmented product classifier.
generator.save("models/product_generator.keras")
augmented_cnn.save("models/augmented_catalog_cnn.keras")


###DAY 12
# Imports the Day 12 tools for text cleaning, TF-IDF and sentiment classification.
import re
import nltk
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Loads the Amazon product review dataset.
reviews_df = pd.read_csv("data/amazon_reviews.csv")

# Downloads and loads English stopwords for review cleaning.
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Cleans reviews by using lowercase text, removing noise and removing stopwords.
def clean_review(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


reviews_df["clean_text"] = reviews_df["text"].apply(clean_review)

# Separates the cleaned reviews and sentiment labels.
X_text = reviews_df["clean_text"]
y_text = reviews_df["label"]

# Splits the reviews into training and testing data.
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
)

# Creates the TF-IDF and Logistic Regression sentiment pipeline.
sentiment_model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Trains the classical sentiment classifier.
sentiment_model.fit(X_train_text, y_train_text)

# Predicts sentiment and evaluates the baseline model.
sentiment_predictions = sentiment_model.predict(X_test_text)

print("Accuracy:", accuracy_score(y_test_text, sentiment_predictions))
print("Precision:", precision_score(y_test_text, sentiment_predictions))
print("Recall:", recall_score(y_test_text, sentiment_predictions))

# Finds the most informative positive and negative review terms.
tfidf = sentiment_model.named_steps["tfidf"]
classifier = sentiment_model.named_steps["classifier"]

feature_names = tfidf.get_feature_names_out()
coefficients = classifier.coef_[0]

positive_terms = feature_names[coefficients.argsort()[-10:][::-1]]
negative_terms = feature_names[coefficients.argsort()[:10]]

print("Positive Terms:", positive_terms)
print("Negative Terms:", negative_terms)

# Saves the Day 12 review-processing and sentiment module.
joblib.dump(sentiment_model, "models/review_sentiment_pipeline.joblib")
