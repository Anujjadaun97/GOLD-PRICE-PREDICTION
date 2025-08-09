### Gold Price Prediction Project README

This README file provides an overview of the Jupyter notebook [Gold_Price_Prediction.ipynb](https://github.com/Anujjadaun97/GOLD-PRICE-PREDICTION/blob/main/Gold_Price_Prediction.ipynb), which focuses on predicting gold prices using a machine learning model.

---

### Project Overview

This project uses a dataset to build a predictive model for gold prices. The notebook outlines a complete machine learning workflow, starting from data collection and preprocessing, moving to model training, and ending with evaluation. The primary objective is to accurately predict the price of gold based on other related financial data.

---

### Key Sections of the Notebook

1.  **Data Collection and Processing**: The notebook loads the [gld_price_data.csv](https://github.com/Anujjadaun97/GOLD-PRICE-PREDICTION/blob/main/gld_price_data.csv) file into a Pandas DataFrame. The initial steps involve loading the data and examining the first and last few rows of the dataset to get a sense of its structure and content. The dataset contains 2290 rows and 6 columns. The columns include 'Date', 'SPX', 'GLD', 'USO', 'SLV', and 'EUR/USD'.
2.  **Data Analysis**: The notebook includes a data analysis section. The data is processed and analyzed to identify patterns and relationships between the variables.
3.  **Train Test Split**: The data is split into training and testing sets to prepare for model training and evaluation.
4.  **Model Training**: A **Random Forest Regressor** model is used for training.
5.  **Evaluation**: The trained model's performance is evaluated using metrics from the `sklearn` library. The notebook imports `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `warnings`, `train_test_split`, `RandomForestRegressor`, and `metrics`.

---

### How to Run the Notebook

To run this notebook, you will need to have the following libraries installed:
* `numpy`
* `pandas`
* `matplotlib.pyplot`
* `seaborn`
* `scikit-learn` (specifically `train_test_split`, `RandomForestRegressor`, and `metrics`)

The notebook requires the [gld_price_data.csv](https://github.com/Anujjadaun97/GOLD-PRICE-PREDICTION/blob/main/gld_price_data.csv) file to be present in the same directory or accessible via the specified path.
