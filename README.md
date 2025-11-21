# Predicting AI Tool Adoption from Developer Survey Data

## Project Overview

This project explores a large developer survey dataset (`survey_results_public.csv`) to understand **which developers are most likely to use AI tools** and why. It consists of exploratory data analysis (EDA), dimensionality reduction (PCA), and a supervised learning model (Random Forest) to predict AI tool adoption from demographic and many career relevant features. 
The key predictive question is:

> Given a developer’s age, years of coding experience, education level, and employment status, how likely are they to use AI tools?

---

## Motivation

AI-powered coding assistants and tools have quickly become part of many developers’ daily workflows. However, some groups embrace AI tools enthusiastically, while others are more cautious.

The goals of this project are to:

- **Describe the developer dataaset and explore the distribution of different features** (age, employment, education, experience) using survey data  
- **Engineer a meaningful target variable** indicating whether a respondent actively uses AI tools. :contentReference
- **Build and evaluate a predictive model** that explains which factors most strongly drive AI tool adoption.
- **Translate technical results into clear, narrative insights** suitable for a non-technical audience.

---

## Dataset

- **Source:** Public “survey_results_public.csv” file from a large annual developer survey (e.g., Stack Overflow Developer Survey).  
- **Size after cleaning for modeling:** 33,012 rows and 168 columns (after dropping rows with missing key variables and one-hot encoding categorical features).

Key columns used in this project include:

- `AISelect` → mapped into **`AI_binary`** (1 = uses AI tools daily/weekly, 0 = does not use AI tools). 
- `Age` → transformed into **`Age_numeric`**, a numeric representation of age ranges.
- `YearsCode` → transformed into **`YearsCode_numeric`**, the approximate number of years of programming experience. 
- `EdLevel` → highest formal education attained. 
- `Employment` → current employment status (e.g., Employed, Student, Independent contractor, Retired).  

---

## Repository Structure

The repository is organized as follows:

- `Survey_Results_Exploratory_Data_Analysis.ipynb`  
  Original Jupyter notebook containing EDA, feature engineering, PCA, and model training code.  

- `survey_results_public.csv`  
  Raw survey data file used as input to the analysis. 

- `README.md`  
  Project documentation (this file), describing motivation, methods, results, and acknowledgements.

---

## Libraries and Tools

The analysis is written in Python and uses the following libraries:

- **Core data & plotting**
  - `pandas` for data loading, cleaning, and manipulation. 
  - `numpy` for numerical operations. 
  - `matplotlib` and `seaborn` for visualizations (boxplots, bar charts, heatmaps, PCA plots, confusion matrix).  

- **Machine learning**
  - `sklearn.model_selection.train_test_split` for splitting training and test sets.
  - `sklearn.preprocessing.StandardScaler` for scaling numeric features (used in PCA and/or pipeline). 
  - `sklearn.decomposition.PCA` for dimensionality reduction and understanding major numeric directions of variation. 
  - `sklearn.ensemble.RandomForestClassifier` as the main predictive model. 
  - `sklearn.metrics` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) for model evaluation.  

- **Environment**
  - Jupyter Notebook for interactive analysis and HTML export. 

---

## Analysis & Methods

1. **Exploratory Data Analysis (EDA)**  
   - Loaded the survey CSV and explored summary statistics with `df.describe(include='all')`.
   - Examined distributions of key variables:
     - **Age** via numeric conversion and boxplots.  
     - **Employment status** via value counts and bar charts (showing most respondents are employed).  

2. **Feature Engineering**
   - Converted age ranges to **`Age_numeric`** and years-of-experience strings to **`YearsCode_numeric`**. 
     - Any “Yes, I use AI tools …” response → 1  
     - “No, I do not use AI tools” → 0 
   - Dropped rows with missing values in key columns (`Age_numeric`, `YearsCode_numeric`, `EdLevel`, `Employment`, `AI_binary`) to obtain `model_df` (33,012 rows).

3. **Model Features**
   - Selected the following modeling features:  
     `['Age_numeric', 'YearsCode_numeric', 'Employment', 'EdLevel']` 
   - Applied one-hot encoding to categorical variables (`Employment`, `EdLevel`), resulting in 14 total model features after encoding. 

4. **Train–Test Split**
   - Split the data into training and test sets using an 80/20 split with a fixed random seed for reproducibility:  
     - Train: 26,409 rows, 14 features  
     - Test: 6,603 rows, 14 features 

5. **Model Training**
   - Trained a **RandomForestClassifier** with:
     - `n_estimators=300`  
     - `random_state=42` 

6. **Evaluation**
   - Computed the following metrics on the held-out test set:
     - **Accuracy:** 0.7790  
     - **Precision:** 0.7988  
     - **Recall:** 0.9628  
     - **F1-score:** 0.8731  
   - Visualized the confusion matrix with a Seaborn heatmap for insight into true/false positives and negatives. 

7. **Feature Importance & PCA**
   - Used the trained Random Forest’s **feature importances** to identify which variables most strongly drove the predictions. Top features:  
     - `YearsCode_numeric` – 0.6342  
     - `Age_numeric` – 0.1678  
     - `Employment_Retired` – 0.0278  
     - `Employment_Independent contractor, freelancer, or self-employed` – 0.0243  
     - `Employment_Not employed` – 0.0230  
     - Multiple `EdLevel_*` categories around ~0.017–0.014  
   - Performed PCA on numeric columns to understand dominant dimensions of variation in the data. 

---

## Summary of Results

- Developers’ **years of coding experience** (`YearsCode_numeric`) and **age** are by far the most important predictors of AI tool usage in this model. 
- Employment categories such as being **Retired**, **Independent contractor/freelancer**, or **Not employed**, as well as education levels (Bachelor’s, Master’s, some college), provide additional but smaller predictive signal.  
- The Random Forest model achieves:
  - ~**78% accuracy**, with a **very high recall (~96%)** for AI users.  
- The confusion matrix shows the model is especially strong at identifying respondents who do use AI tools (few false negatives), which is useful if the goal is to find potential early adopters for programs or products. 

---

## How to Run

1. **Clone this repository** and ensure `survey_results_public.csv` is present in the root directory (or update the path in the notebook).
2. **Install dependencies**, for example:


