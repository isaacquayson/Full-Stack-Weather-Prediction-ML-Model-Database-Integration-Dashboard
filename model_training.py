# importing the needed libraries
import pandas as pd
import numpy as np
import pickle
import warnings

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier  # Using Gradient Boosting

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("C://Users//quays//Desktop//weather_classification_data.csv")

# Removing outliers function
def remove_outliers(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    capped_df = df.copy()

    for col in num_cols:
        Q1 = capped_df[col].quantile(0.25)
        Q3 = capped_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        capped_df[col] = capped_df[col].clip(lower=lower_bound, upper=upper_bound)

    return capped_df

# Apply outlier removal
cleaned_df = remove_outliers(df)

# Convert target variable to numerical labels
label_encoder = LabelEncoder()
cleaned_df['weather_type_encoded'] = label_encoder.fit_transform(cleaned_df['weather_type'])

# Train/Validation/Test Split
train_validation_data, test_data = train_test_split(cleaned_df, test_size=0.2, random_state=43)
train_data, validation_data = train_test_split(train_validation_data, test_size=0.25, random_state=47)

# Features and Target
feature_cols = ['temperature','humidity','wind_speed','precipitation(%)',
                'cloud_cover','atmospheric_pressure','uv_index','season',
                'visibility(km)','location']
target_col = 'weather_type_encoded'  # Use the encoded target

train_X = train_data[feature_cols]
train_y = train_data[target_col]

val_X = validation_data[feature_cols]
val_y = validation_data[target_col]

test_X = test_data[feature_cols]
test_y = test_data[target_col]

# Preprocessing Setup
categorical_cols = ["cloud_cover", "season", "location"]
numeric_cols = [col for col in feature_cols if col not in categorical_cols]

scaler = MinMaxScaler()
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit Transformers on Training Data
# Scale numeric
scaled_numeric = scaler.fit_transform(train_X[numeric_cols])

# Encode categorical
encoded_categorical = encoder.fit_transform(train_X[categorical_cols])

# Combine processed training data
train_X_processed = np.hstack([scaled_numeric, encoded_categorical])

# Validation
val_scaled_numeric = scaler.transform(val_X[numeric_cols])
val_encoded_categorical = encoder.transform(val_X[categorical_cols])
val_X_processed = np.hstack([val_scaled_numeric, val_encoded_categorical])

# Test
test_scaled_numeric = scaler.transform(test_X[numeric_cols])
test_encoded_categorical = encoder.transform(test_X[categorical_cols])
test_X_processed = np.hstack([test_scaled_numeric, test_encoded_categorical])

# Model Training
model = XGBClassifier(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(train_X_processed, train_y)

# Evaluation
train_pred = model.predict(train_X_processed)
val_pred = model.predict(val_X_processed)
test_pred = model.predict(test_X_processed)

# Convert predictions back to original labels for reporting
train_pred_labels = label_encoder.inverse_transform(train_pred)
val_pred_labels = label_encoder.inverse_transform(val_pred)
test_pred_labels = label_encoder.inverse_transform(test_pred)

# Get original target values for reporting
train_y_labels = label_encoder.inverse_transform(train_y)
val_y_labels = label_encoder.inverse_transform(val_y)
test_y_labels = label_encoder.inverse_transform(test_y)

print("Training Accuracy:", accuracy_score(train_y_labels, train_pred_labels))
print("Validation Accuracy:", accuracy_score(val_y_labels, val_pred_labels))
print("Test Accuracy:", accuracy_score(test_y_labels, test_pred_labels))
print("\nClassification Report (Test):\n", classification_report(test_y_labels, test_pred_labels))
print("Confusion Matrix (Test):\n", confusion_matrix(test_y_labels, test_pred_labels))

print((pd.Series(train_y_labels).value_counts()/len(train_y_labels))*100)

# Save Model, Scaler, Encoder, and Label Encoder
pickle.dump(model, open('xgb_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))

print("\n✅ Model, Scaler, Encoder, and Label Encoder saved successfully!")