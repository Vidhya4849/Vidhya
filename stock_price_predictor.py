# 1. Load the dataset
import pandas as pd
import numpy as np # Ensure numpy is imported for np.number

# Adjust the path if needed
df = pd.read_csv('/content/ai_stock_market_time_series (2).csv')
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 2. Basic EDA
print(df.info())
print(df.describe())
print("Missing values before handling:\n", df.isnull().sum()) # Print missing values before handling

# 3. Handle Missing Values
# Identify numerical columns (excluding the target and potentially 'Date')
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude the target if it's in numerical_cols
if 'Close' in numerical_cols:
    numerical_cols.remove('Close')

# Impute missing values in numerical columns with the mean
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# For the target variable 'Close', you might choose to drop rows with missing values
# or impute them. Dropping is often safer for the target.
df.dropna(subset=['Close'], inplace=True)

print("Missing values after handling:\n", df.isnull().sum()) # Verify missing values are handled

# 4. Visualize key features
import seaborn as sns
import matplotlib.pyplot as plt

# Example: visualize closing price if exists
if 'Close' in df.columns:
    sns.histplot(df['Close'], kde=True)
    plt.title("Distribution of Closing Prices")
    plt.show()

# 5. Feature Selection
target = 'Close'  # You can change this
# Exclude 'Date' and the target column from features for this Linear Regression model
features = df.columns.drop(['Date', target])
print("Features:", features)

# 6. Handle Categorical Features (assuming 'Date' was the only non-numeric non-target one intended)
# Based on your original code and EDA, 'Date' was the only column identified as object dtype.
# If there were other categorical columns, you would handle them here.
categorical_cols = df.select_dtypes(include='object').columns
print("Categorical Columns:", categorical_cols)

# Since we excluded 'Date' from features, no categorical encoding needed for this model setup.
# If 'Date' was crucial as a feature in a different format (e.g., year, month), it would need transformation.
# For this example, we proceed without encoding 'Date'.
df_processed = df.drop('Date', axis=1).copy()


# 7. Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Scale features (excluding the target)
X_scaled = scaler.fit_transform(df_processed.drop(target, axis=1))
y = df_processed[target]

# 8. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. Train Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 10. Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 11. Deploy with Gradio
!pip install gradio --quiet

import gradio as gr # Ensure gradio is imported here

# Note: The Gradio interface setup needs to align with the features used for training.
# Since we excluded 'Date' from features in the Linear Regression,
# the Gradio interface should only request inputs for the numerical features.
# If you intend to use 'Date' or other categorical features in a different model,
# the Gradio setup would need adjustment.

# Rebuild Gradio input list dynamically based on the features used for training
input_components = []
# Get the list of feature columns after processing and before scaling
processed_features = df_processed.drop(target, axis=1).columns
for col in processed_features:
     # Assuming all remaining features are numerical after dropping 'Date' and handling NaNs
     input_components.append(gr.Number(label=col))

output_component = gr.Number(label=f"📈 Predicted {target}")

# The predict_stock function needs to handle input based on the *trained model's features*.
# It also needs to apply the same scaler.
# It assumes the input dictionary will contain values for the processed numerical features.
def predict_stock(**kwargs):
    # Create a DataFrame from input, aligning columns with training features
    input_df = pd.DataFrame([kwargs])

    # Ensure the input DataFrame has the same columns as the features used for training,
    # in the correct order, filling missing ones with 0 (or other appropriate value if needed)
    # This step is crucial for the scaler and model prediction.
    input_df = input_df.reindex(columns=processed_features, fill_value=0)


    # Scale the input using the same scaler used for training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    return round(prediction[0], 2)


# Gradio Interface setup using the dynamically created input components
gr.Interface(
    fn=predict_stock,
    inputs=input_components,
    outputs=output_component,
    title="📊 Stock Price Predictor",
    description=f"Enter stock data features to predict the {target}."
).launch()