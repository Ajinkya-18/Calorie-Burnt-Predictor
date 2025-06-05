# 🏃‍♂️ Calories Burnt Predictor

This project predicts the number of calories burnt during physical activity based on various physiological parameters using machine learning regression models.

## 📂 Dataset

- **Source:** [Kaggle - fmendesdat263xdemos] (https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)
- **Files Used:**
  - `exercise.csv`: Contains exercise-related physiological features.
  - `calories.csv`: Contains calorie values for each exercise session.

These datasets were combined and preprocessed for training regression models.

## 🧠 Features Used

| Feature        | Description                         |
|----------------|-------------------------------------|
| `Age`          | Age of the person (years)           |
| `Height`       | Height (in centimeters)             |
| `Weight`       | Weight (in kilograms)               |
| `Duration`     | Duration of exercise (in minutes)   |
| `Heart_Rate`   | Average heart rate during exercise  |
| `Body_Temp`    | Average body temperature (°C)       |
| `Calories`     | 🔥 Target variable (calories burnt) |

> Optional feature engineering includes computing **BMI** from Height and Weight to reduce collinearity.

## 🛠️ ML Techniques Used

- **Exploratory Data Analysis (EDA)**
  - Scatter plots and correlation heatmaps
  - Multicollinearity inspection among features
- **Feature Engineering**
  - BMI calculation from height and weight
- **Preprocessing**
  - Label encoding for any categorical columns (if present)
  - Train-test split
  - Feature scaling
- **Regression Models**
  - Linear Regression
  - Lasso Regression
  - Ridge Regression (for regularization)
  - Decision Tree Regressor
  - Random Forest Regressor
  - K Neighbors Regressor
  - (Optional) XGBoost or other ensemble regressors

## 📈 Evaluation Metrics

- **R² Score**
- **Mean Absolute Error (MAE)**

## 🧪 Results & Insights

- `Duration`, `Heart_Rate`, and `Body_Temp` showed strong correlation with calories burnt.
- Multicollinearity was addressed using regularization and potential PCA.
- Tree-based models (like Random Forest) performed better due to their robustness to feature correlation.

## 📊 Visualization

- Feature vs. Calories scatter plots
- Model prediction vs. actual calories
- Correlation heatmaps

## 🚀 How to Run

### Option 1: Run with `conda` (recommended)

Create and activate the environment from `environment.yml`:
  conda env create -f environment.yml
  conda activate calorie-predictor

### Option 2: Run with pip

Clone the repository:
git clone https://github.com/Ajinkya-18/Calorie-Burnt-Predictor.git
cd Calorie-Burnt-Predictor

Install dependencies:
pip install -r requirements.txt


## 🏁 Conclusion
This project demonstrates the use of supervised learning for predicting calorie expenditure based on physiological parameters. It highlights the importance of feature selection, handling multicollinearity, and choosing the right model for better performance.

## Acknowledgements
#### Dataset credit: Fernando Mendes on Kaggle

#### Implemented by: Ajinkya Tamhankar
