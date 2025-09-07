# 📊 Bank Marketing Campaign Analysis with Machine Learning

## 🔍 Project Description
The goal of this project was to build a machine learning model to predict whether a bank client will subscribe to a term deposit after a marketing contact.  
The dataset includes demographic features, historical contact information, and macroeconomic indicators.

## 🛠️ Tools and Libraries
- Python (pandas, numpy, matplotlib, seaborn)  
- scikit-learn (classification models, metrics)  
- XGBoost, LightGBM, GradientBoosting  
- SHAP (feature impact analysis)  

## 📈 Workflow
1. **Data Preprocessing**  
   - Data cleaning and feature engineering (e.g., `campaign_group`), categorization.  
   - Scaling and preparation of training and validation sets.

2. **Model Training**  
   - Tested models: Logistic Regression, KNN, Decision Tree, Gradient Boosting, XGBoost, LightGBM.  
   - Hyperparameter optimization using GridSearch and Hyperopt.

3. **Model Evaluation**  
   - Main metrics: F1-score, AUROC.  
   - Class imbalance carefully considered.  
   - The **GradientBoostingClassifier** achieved the best performance (AUROC ≈ 0.81, F1_val ≈ 0.37).

4. **Model Interpretation (SHAP)**  
   - Top influential features:  
     - `nr.employed` – employment rate in the economy  
     - `contact_telephone` – contact channel  
     - `euribor3m` – 3-month Euribor interest rate  
     - `cons.conf.idx` – consumer confidence index  
     - `age` – client age

5. **Error Analysis**  
   - The model accurately predicts class “0” (no subscription) but struggles with class “1” (subscription).  
   - A significant number of **false negatives**: clients who subscribed but were predicted as non-subscribers.

## 📊 Results

![Alt text](https://github.com/Alenushka2013/Bank-Marketing-Dataset-Machine-Learning-Project/blob/main/Results_of_experiments.jpg)

- Accuracy: **90%**  
- F1-score for class “1”: **0.37**  
- Confusion matrix:  
  |               | Predicted 0 | Predicted 1 |
  |---------------|------------|------------|
  | Actual 0      | 7151       | 157        |
  | Actual 1      | 684        | 244        |

## 💡 Conclusions
- Macroeconomic indicators are the most influential factors for campaign outcomes.  
- Contact channel and client age are important features for targeting.  
- The model is biased towards predicting non-subscription more often than subscription.

## 🚀 Recommendations
1. **Model Improvements**  
   - Apply class balancing techniques (SMOTE, class weights).  
   - Consider ensemble approaches (e.g., stacking).  
   - Optimize for recall on class “1” to reduce missed opportunities.

2. **Business Recommendations**  
   - Personalize communication channels (different strategies for phone vs email).  
   - Consider macroeconomic context when launching campaigns.  
   - Segment clients by age to increase offer relevance.
