import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('salary_data.csv')  # Replace with the actual dataset path

# Display basic information about the dataset
print(df.info())

# Data Preprocessing
# Handling missing values (drop rows with missing salary)
df = df.dropna(subset=['lowest_salary', 'highest_salary', 'country_name', 'continent_name', 'wage_span'])

# Fill missing values for other fields
df['wage_span'] = df['wage_span'].fillna('Unknown')
df['continent_name'] = df['continent_name'].fillna('Unknown')

# Feature engineering: Create a target column 'average_salary'
df['average_salary'] = (df['lowest_salary'] + df['highest_salary']) / 2

# Encode categorical features using separate LabelEncoders
country_encoder = LabelEncoder()
continent_encoder = LabelEncoder()
wage_span_encoder = LabelEncoder()

df['country_encoded'] = country_encoder.fit_transform(df['country_name'])
df['continent_encoded'] = continent_encoder.fit_transform(df['continent_name'])
df['wage_span_encoded'] = wage_span_encoder.fit_transform(df['wage_span'])

# Selecting features for the model
X = df[['country_encoded', 'continent_encoded', 'wage_span_encoded']]
y = df['average_salary']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Test the model with new input data
def predict_salary(country_name, continent_name, wage_span):
    # Encode input data
    country_encoded = country_encoder.transform([country_name])[0]
    continent_encoded = continent_encoder.transform([continent_name])[0]
    wage_span_encoded = wage_span_encoder.transform([wage_span])[0]

    # Prepare input for prediction
    input_data = [[country_encoded, continent_encoded, wage_span_encoded]]
    predicted_salary = model.predict(input_data)[0]
    
    return predicted_salary

# Example usage
country_name = 'Albania'
continent_name = 'Europe'
wage_span = 'Monthly'
predicted_salary = predict_salary(country_name, continent_name, wage_span)

print(f"Predicted Salary in {country_name}, {continent_name}, with a {wage_span} wage span: ${predicted_salary:.2f}")