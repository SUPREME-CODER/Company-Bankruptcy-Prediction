## Company Bankruptcy Prediction

### Problem Statement
A failing company or business can have far-reaching consequences, not only impacting the country's economy but also affecting the personal and professional lives of its employees. This highlights the critical importance of predicting potential downfalls in advance, enabling investors, stakeholders, and decision-makers to take timely and effective measures to mitigate risks and facilitate recovery.

This project aims to analyze various aspects, features, and key performance indicators (KPIs) of companies to develop a predictive model. The model will be designed to learn from historical data and identify patterns indicative of a potential business decline. By providing early alerts, this solution seeks to empower investors and stakeholders with actionable insights to safeguard their interests and support strategic interventions.

[Data Source](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

The Data included a total of 95 features including :-
* After-tax net Interest Rate  &rarr; Net Income/Net Sales
* Non-industry income and expenditure/revenue  &rarr; Net Non-operating Income Ratio
* Operating Expense Rate  &rarr; Operating Expenses/Net Sales
* Research and development expense rate  &rarr; (Research and Development Expenses)/Net Sales
* Cash flow rate: Cash Flow from Operating/Current Liabilities
* Effective Tax Rate
* Net Value Per Share
* Cash Flow Per Share
* Total Asset Turnover
* Net profit before tax
* Revenue per person
* Total income/Total expense
* Total expense/Assets
* Cash Flow to Sales
* Gross Profit to Sales
* Net Income Flag
* Equity to Liability

The data was obtained from UCI Machine Learning Repository: [Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction)

These features describe the overall financial situation of the company using which we will figure out the exact condition of the company and predict if the company will go bankrupt in the near future or not.

## Dependencies
Tech Stack used - XGBClassifier, GradientBoostingClassifier, Borderline-SMOTE, Pandas, Seaborn, Matplotlib.

## Project Workflow
* **Data Quality Check** - Performed some data quality checks which included finding out the shape of the data, knowing which ones are date columns, categorical and numerical features. For CF their number of unique values, their value counts etc.
* **Feature Selection**
    * **Mutual Information** - Mutual Information is a measure of how much information you can get of one variable by observing another variable. Used this to figure out which features were capable of giving more information about our target variable. And created a set of those features.
    * **Predictive Power Score** - The PPS is an asymmetric, data-type-agnostic score that can detect linear or non-linear relationships between two columns. The score ranges from 0 (no predictive power) to 1 (perfect predictive power). Created another set of features which had non-zero PPS for the target variable.
    * **Feature Importances - RandomForestClassifier** - Used the simple feature importance calculation for all the features.
    * **Feature Importances - DecisionTreeClassifier** - Used the simple feature importance calculation for all the features.
    * **Hypothesis Testing** - Performed 2 sample T-Test for finding out if there was a significant difference between the feature values for bankrupt and non-bankrupt companies. Finally created another set of features and combined all the set of features and acquired 27 features which had the most amount of information and predictive power for the target variable. 
* **Some Viz** - Tried finding out if the features had some outliers or not since these outliers will help us give more information about the bankrupt companies.
* **Balancing Data using BorderlineSMOTE** - Used for balancing the unbalanced classes. Used Borderline since there wansn't much difference between the data points give from the previous vizzes as there wasn't much difference betweent their averages.So assumed that the data might be crowded near the decision boundary and thus Border line SMOTE would have been more helpful to capture the corner cases for better decision boundary classification.
* **Outliers by Local Outlier Factor** - Used as a feature since bankruptcy isn't as common in the dataset like successful / non-bankrupt companies. So used LOF to findout which ones are an outlier and use that as another piece of information.
* **PCA** - Even getting so many important fetures the dimensionaltiy of the data is too high and can affect the training of model so applying PCA to reduce the data while retaining as much information as we can. The PCA did not improve the model training performance. So did not use the reduced data.
* **Model Building using GridSearchCV** - used XGBClassifier and GradientBoostingClassifier, since these algorithms provide the benefits of Boosting and other features of tree algorithms, fast and optimizations.
* **Best Model** - **XGBClassifier** performed better as it performed well on all metrics of accuracy, precision, recall and f1score.


## Usage
Install the Jupyter Notebook and access the data mentioned in the link in Data Source and execute the cells one by one.

## Results
Provide the Final Insights section
* The models did a good job predicting the banruptcy of a company, and the model we have selected is the XGBClassifier as the best model.
* We could have had more info on the companies as the attrition rate in the past few years, declining number of employees indicate that the company is cutting their cost to save the company indicating a bankruptcy, number of projects acquired by the company etc.


## Final Notes
This is a great project to learn:-
* Handling of an imbalanced dataset.
* Working with high dimensional data and figuring out which features are the best or most suited ones.
* Perform outlier detection and use it as a feature.

