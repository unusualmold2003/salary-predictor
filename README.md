# Salary Prediction App

## Project Overview
This Salary Prediction App utilizes a linear regression model to predict the average salary for a given country, continent, and wage span (e.g., monthly, yearly). The project makes use of a dataset that contains salary information from various countries and regions, including the lowest, highest, median, and average salaries.

## Dataset
The dataset used in this project includes the following columns:
- **country_name**: Name of the country.
- **continent_name**: Name of the continent.
- **wage_span**: The salary wage span (e.g., Monthly, Yearly).
- **median_salary**: Median salary for that country.
- **average_salary**: Average salary for that country.
- **lowest_salary**: Lowest salary recorded.
- **highest_salary**: Highest salary recorded.

The dataset was cleaned and preprocessed to handle missing values, and categorical data was encoded for use in the regression model.

You can access the dataset from [Kaggle: LinkedIn Job Postings Dataset]([https://www.kaggle.com/datasets/zedataweaver/global-salary-data]).

## Features
- Linear Regression model for salary prediction.
- Categorical encoding of country, continent, and wage span.
- Predicts the average salary based on country, continent, and wage span.
- Basic evaluation metrics such as MAE (Mean Absolute Error), MSE (Mean Squared Error), and RMSE (Root Mean Squared Error).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/salary-predictor.git
2. Go to the working directory
   ```bash
   cd salary-predictor
3. Install the required packages
   ```bash
   pip install scikit-learn pandas --upgrade
4. Run the app
   ```bash
   python main.py

## Usage
The app takes three input parameters:

  country_name: e.g., 'Albania'
  continent_name: e.g., 'Europe'
  wage_span: e.g., 'Monthly'
  
It predicts the average salary for the given inputs. Here's an example of how you can use it in the code:

```bash
country_name = 'Albania'
continent_name = 'Europe'
wage_span = 'Monthly'
predicted_salary = predict_salary(country_name, continent_name, wage_span)
print(f"Predicted Salary: ${predicted_salary:.2f}")
```

## Model Evaluation
  Mean Absolute Error (MAE): 3385.98
  Mean Squared Error (MSE): 18416682.04
  Root Mean Squared Error (RMSE): 4291.47

## Future Improvements
  Integrate more advanced models like Random Forest or XGBoost for better accuracy.
  Incorporate additional features such as job title, industry, or experience level.
  Add a web interface using Flask or Django.
