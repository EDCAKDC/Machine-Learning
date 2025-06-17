from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import shap
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xgboost import XGBClassifier
from missforest import MissForest
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('data/diabetes.csv', encoding="GBK")
print(data.head())
print(data.info())

# Count of outcome values
print("Number of each outcome label:", data['Outcome'].value_counts())
print("Proportion of each outcome label:",
      data['Outcome'].value_counts(normalize=True))

# List of continuous variables
vars_to_plot = ['Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Count of missing values for each variable
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Save histograms to the plot folder
for col in vars_to_plot:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'plot/{col}_hist.png')
    plt.close()

# Check and fix invalid BloodPressure values (e.g., = 0)
print("Count of BloodPressure == 0:", (data['BloodPressure'] == 0).sum())
bp_median = data.loc[data['BloodPressure'] != 0, 'BloodPressure'].median()
data['BloodPressure'] = data['BloodPressure'].replace(0, bp_median)
print("Count of abnormal values (=0) after correction:",
      (data['BloodPressure'] == 0).sum())

# Plot fixed BloodPressure distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['BloodPressure'], kde=True)
plt.title("Fixed Distribution of BloodPressure")
plt.tight_layout()
plt.savefig("plot/BloodPressure_fixed.png")
plt.close()

# Check and fix invalid BMI values
print("Count of BMI == 0:", (data['BMI'] == 0).sum())
BMI_median = data.loc[data['BMI'] != 0, 'BMI'].median()
data['BMI'] = data['BMI'].replace(0, BMI_median)
print("Count of abnormal values (=0) after correction:",
      (data['BMI'] == 0).sum())

# Plot fixed BMI distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['BMI'], kde=True)
plt.title("Fixed Distribution of BMI")
plt.tight_layout()
plt.savefig("plot/BMI_fixed.png")
plt.close()
print(data['BMI'].describe())

# Check and fix invalid Glucose values
print("Count of Glucose == 0:", (data['Glucose'] == 0).sum())
GLU_median = data.loc[data['Glucose'] != 0, 'Glucose'].median()
data['Glucose'] = data['Glucose'].replace(0, GLU_median)
print("Count of abnormal values (=0) after correction:",
      (data['Glucose'] == 0).sum())

# Plot fixed Glucose distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['Glucose'], kde=True)
plt.title("Fixed Distribution of Glucose")
plt.tight_layout()
plt.savefig("plot/Glucose_fixed.png")
plt.close()
print(data['Glucose'].describe())

# Fix Insulin values
insulin_median = data.loc[data['Insulin'] != 0, 'Insulin'].median()
data['Insulin'] = data['Insulin'].replace(0, insulin_median)
plt.figure(figsize=(6, 4))
sns.histplot(data['Insulin'], kde=True)
plt.title("Fixed Distribution of Insulin")
plt.tight_layout()
plt.savefig("plot/Insulin_fixed.png")
plt.close()
print(data['Insulin'].describe())

# Fix SkinThickness values
ST_median = data.loc[data['SkinThickness'] != 0, 'SkinThickness'].median()
data['SkinThickness'] = data['SkinThickness'].replace(0, ST_median)
plt.figure(figsize=(6, 4))
sns.histplot(data['SkinThickness'], kde=True)
plt.title("Fixed Distribution of SkinThickness")
plt.tight_layout()
plt.savefig("plot/SkinThickness_fixed.png")
plt.close()
print(data['SkinThickness'].describe())

# Save cleaned data
data.to_csv("data/data_imputed.csv", index=False)
#####################################################################################
#################################### Data Splitting #################################
#####################################################################################

# 1. Split the dataset into training and testing sets
data = pd.read_csv("data/data_imputed.csv",
                   encoding="GBK")  # Load imputed data
train_data, test_data = train_test_split(data, test_size=0.3,
                                         stratify=data["Outcome"], random_state=2025)

# Save train and test datasets
train_data.to_csv("data/train_data_notscaled.csv", index=False)
test_data.to_csv("data/test_data_notscaled.csv", index=False)

# 2. Check feature distribution balance between train and test sets
train_data["group"] = "train_set"
test_data["group"] = "test_set"
total = pd.concat([train_data, test_data])

# Create descriptive statistics table
categorical_vars = ["Outcome"]
all_vars = total.columns.values[0:len(total.columns)-1].tolist()
varbalance_table = TableOne(data=total, columns=all_vars,
                            categorical=categorical_vars, groupby="group", pval=True)

# Save balance table
varbalance_table.to_csv("table/varbalance_table.csv", encoding="utf-8-sig")

#####################################################################################
#################################### Feature Engineering ############################
#####################################################################################

# 1. Standardize continuous variables to speed up ML model convergence
continuous_vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Standardize train set
train_data = train_data.drop(columns='group')
train_data[continuous_vars] = StandardScaler(
).fit_transform(train_data[continuous_vars])
train_data.to_csv("data/train_data_scaled.csv", index=False)

# Standardize test set
test_data = test_data.drop(columns='group')
test_data[continuous_vars] = StandardScaler(
).fit_transform(test_data[continuous_vars])
test_data.to_csv("data/test_data_scaled.csv", index=False)

#####################################################################################
####################### 2. Feature Selection Using Logistic and LASSO ###############
#####################################################################################

# --- Method 1: Logistic Regression-Based Variable Selection ---
train_data = pd.read_csv("data/train_data_notscaled.csv", index_col=0)
all_vars = train_data.columns.values.tolist()
independent_vars = all_vars[:-1]
y_train = train_data['Outcome']

# Univariate logistic regression
results_univariable = []
for var in independent_vars:
    print("####### " + var + " ########")
    x = sm.add_constant(train_data[var])
    model = sm.Logit(y_train, x).fit(disp=0)
    coef = model.params[var]
    p_value = model.pvalues[var]
    results_univariable.append(
        {'Variable': var, 'Coefficient': coef, 'P-value': p_value})

# Save univariate results
results_univariable_df = pd.DataFrame(results_univariable)
significant_vars_univ = results_univariable_df[results_univariable_df['P-value']
                                               < 0.05]['Variable'].tolist()

# Multivariate logistic regression using significant variables
X_sig = train_data[significant_vars_univ]
model_multilog = sm.Logit(y_train, X_sig).fit(disp=0)
results_mulvariable_df = pd.DataFrame({
    "Variable": X_sig.columns,
    "Coefficient": model_multilog.params,
    "Odd Ratio": np.exp(model_multilog.params),
    "P-value": model_multilog.pvalues
})
results_mulvariable_df.to_csv("table/results_mulvariable_df.csv", index=False)
significant_vars_multi = results_mulvariable_df[results_mulvariable_df['P-value']
                                                < 0.05]['Variable'].tolist()

# Save significant variable list
with open('file/significant_vars.pkl', 'wb') as f:
    pickle.dump(significant_vars_multi, f)

# --- Method 2: LASSO Regression-Based Variable Selection ---
# Lasso regression (alpha controls penalty strength)
lasso = Lasso(alpha=0.1)
X_train = train_data[independent_vars]
lasso.fit(X_train, y_train)

# Select non-zero coefficient variables
selected_variables = np.array(independent_vars)[lasso.coef_ != 0].tolist()
###################################################################################################################
########### Build predictive models on training data and tune hyperparameters using validation set ###############
########### (Test set should only be used for final evaluation) ##################################################
###################################################################################################################

############################# 1. Logistic Regression Model ##############################

# Load significant variables selected from multivariate logistic regression
with open('file/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)

train_data = pd.read_csv("data/train_data_notscaled.csv",
                         encoding="GBK", index_col=0)
X_train = train_data[significant_vars_multi]
X_train_const = sm.add_constant(X_train)
y_train = train_data['Outcome']

# Train logistic regression model
logist_model = sm.Logit(y_train, X_train_const).fit(disp=0)
logist_model.summary()

# Evaluate model on training set using AUC
y_train_pred_prob_logist = logist_model.predict(X_train_const)
auc_logist = roc_auc_score(y_train, y_train_pred_prob_logist)
print("Logistic model training AUC:", auc_logist)

# Save trained logistic regression model
with open("model/logistic_model.pkl", 'wb') as f:
    pickle.dump(logist_model, f)

############################### 2. Decision Tree Model ##################################

# Load standardized training data
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="GBK", index_col=0)
X = train_data_scaled.drop(columns='Outcome')
y = train_data_scaled['Outcome']

# Split into internal training and validation sets (70% / 30%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

# 2.2. Train decision tree with default parameters
tree_default = DecisionTreeClassifier(random_state=123)
tree_default.fit(X_train, y_train)

# Check default parameters
print("Default model parameters:\n", pd.DataFrame.from_dict(
    tree_default.get_params(), orient='index'))

# 2.3. Evaluate default model on validation set
y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1]
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed)
print("Validation AUC of default model:", auc_treed)

# 2.4. Define parameter search space for RandomizedSearchCV
param_distributions = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [5, 10, 20],
    'max_features': ['sqrt', None],
    'ccp_alpha': [0.0, 0.01, 0.1]
}

# 2.5. Perform randomized search for hyperparameter tuning
tree_model = DecisionTreeClassifier(random_state=123)
random_search = RandomizedSearchCV(
    estimator=tree_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='roc_auc',
    n_jobs=-1,
    cv=3,
    verbose=1,
    random_state=123
)
random_search.fit(X_train, y_train)

# 2.6. Retrieve best model and parameters
tree_model_best = random_search.best_estimator_
best_params = random_search.best_params_

# 2.7. Evaluate best model on validation set
y_val_pred_prob = tree_model_best.predict_proba(X_val)[:, 1]
best_auc_tree = roc_auc_score(y_val, y_val_pred_prob)
print("Best parameter combination:", best_params)
print("Validation AUC of default model:", auc_treed)
print("Validation AUC after tuning:", best_auc_tree)
print("Best model parameters:\n", pd.DataFrame.from_dict(
    tree_model_best.get_params(), orient='index'))

# 2.8. Plot and save the decision tree
plt.figure(figsize=(10, 5))
plot_tree(tree_model_best, feature_names=X.columns,
          class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.savefig("plot/tree_structure.jpg", dpi=500)
plt.show()

# Save the trained tree model
with open("model/tree_model.pkl", 'wb') as f:
    pickle.dump(tree_model_best, f)
########################## 3. Random Forest (RF) Model ##########################
# Train default RF model
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True)
rf_model_default.fit(X_train, y_train)

# Print default parameters
print("Default model parameters:", pd.DataFrame.from_dict(
    rf_model_default.get_params(), orient='index'))

# Evaluate AUC on validation set
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("Validation AUC (default RF):", auc_rfd)

# Define hyperparameter search space
param_distributions = {
    'n_estimators': np.arange(50, 500, 50),
    'max_features': list(range(2, round(np.sqrt(X.shape[1])) + 1))
}

# RandomizedSearchCV for tuning
rf_model = RandomForestClassifier(random_state=123, n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=123
)
random_search.fit(X_train, y_train)

# Best model and evaluation
rf_model_best = random_search.best_estimator_
best_params_rf = random_search.best_params_
y_val_pred_prob = rf_model_best.predict_proba(X_val)[:, 1]
best_auc_rf = roc_auc_score(y_val, y_val_pred_prob)

print("Best RF parameters:", best_params_rf)
print("Validation AUC (tuned RF):", best_auc_rf)
print("Tuned RF model parameters:", pd.DataFrame.from_dict(
    rf_model_best.get_params(), orient='index'))

# Save model
with open("model/rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)


########################## 4. XGBoost Model ##########################
xgb_default = XGBClassifier(
    random_state=123, use_label_encoder=False, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("Validation AUC (default XGBoost):", auc_xgbd)

param_distributions = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0]
}

xgb_model = XGBClassifier(
    tree_method='hist',
    predictor='cpu_predictor',
    random_state=123,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=123
)
random_search.fit(X_train, y_train)
xgb_model_best = random_search.best_estimator_
best_params_xgb = random_search.best_params_
y_val_pred_prob = xgb_model_best.predict_proba(X_val)[:, 1]
best_auc_xgb = roc_auc_score(y_val, y_val_pred_prob)

print("Best XGBoost parameters:", best_params_xgb)
print("Validation AUC (tuned XGBoost):", best_auc_xgb)
print("Tuned XGBoost model parameters:", pd.DataFrame.from_dict(
    xgb_model_best.get_params(), orient='index'))

with open("model/xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)


########################## 5. LightGBM Model ##########################
lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("Validation AUC (default LightGBM):", auc_lgbd)

param_distributions = {
    'learning_rate': [0.05],
    'num_leaves': [15, 31],
    'n_estimators': [50],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

lgb_model = LGBMClassifier(
    random_state=123,
    n_jobs=1,
    verbose=-1,
    min_data_in_leaf=20,
    max_depth=3
)

random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_distributions,
    n_iter=2,
    scoring='roc_auc',
    cv=2,
    n_jobs=1,
    random_state=123
)
random_search.fit(X_train, y_train)
lgb_model_best = random_search.best_estimator_
best_params_lgb = random_search.best_params_
y_val_pred_prob = lgb_model_best.predict_proba(X_val)[:, 1]
best_auc_lgb = roc_auc_score(y_val, y_val_pred_prob)

print("Best LightGBM parameters:", best_params_lgb)
print("Validation AUC (tuned LightGBM):", best_auc_lgb)

with open("model/lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)
###################### 6. Support Vector Machine (SVM) Model ##########################
# 6.1. Train default SVM model
svm_default = SVC(probability=True, random_state=123)
svm_default.fit(X_train, y_train)

# 6.2. Evaluate default model on validation set
y_val_pred_prob_svmd = svm_default.predict_proba(X_val)[:, 1]
auc_svmd = roc_auc_score(y_val, y_val_pred_prob_svmd)
print("Validation AUC (default SVM):", auc_svmd)

# 6.2. Define search space for random search (simplified for memory efficiency)
param_distributions = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01],
    'degree': [2, 3]
}

# 6.3. Random search setup
svm_model = SVC(probability=True, random_state=123)
random_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=param_distributions,
    n_iter=8,
    scoring='roc_auc',
    cv=2,
    random_state=123,
    verbose=1,
    n_jobs=1
)
random_search.fit(X_train, y_train)

# 6.5. Evaluate best model
svm_model_best = random_search.best_estimator_
best_params_svm = random_search.best_params_
y_val_pred_prob = svm_model_best.predict_proba(X_val)[:, 1]
best_auc_svm = roc_auc_score(y_val, y_val_pred_prob)

print("Best SVM parameters:", best_params_svm)
print("Validation AUC (tuned SVM):", best_auc_svm)
print("Parameter comparison:")
print(pd.concat(
    [pd.DataFrame.from_dict(svm_default.get_params(), orient='index', columns=['Default']),
     pd.DataFrame.from_dict(svm_model_best.get_params(), orient='index', columns=['Tuned'])],
    axis=1))

# 6.7. Save model
with open("model/svm_model.pkl", 'wb') as f:
    pickle.dump(svm_model_best, f)

###################### 7. Artificial Neural Network (ANN) Model ##########################
# 7.1. Train default ANN model
ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)

# 7.2. Evaluate default ANN model
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("Validation AUC (default ANN):", auc_annd)

# 7.2. Define search space (simplified)
param_distributions = {
    'hidden_layer_sizes': [(25,), (50,), (50, 50)],
    'activation': ['relu', 'tanh']
}

# 7.3. Create base model
ann_model = MLPClassifier(random_state=123, max_iter=500)

# 7.4. Random search
random_search = RandomizedSearchCV(
    estimator=ann_model,
    param_distributions=param_distributions,
    n_iter=4,
    scoring='roc_auc',
    cv=2,
    n_jobs=1,
    verbose=1,
    random_state=123
)
random_search.fit(X_train, y_train)

# 7.6. Evaluate best ANN model
ann_model_best = random_search.best_estimator_
best_params_ann = random_search.best_params_
y_val_pred_prob_ann = ann_model_best.predict_proba(X_val)[:, 1]
best_auc_ann = roc_auc_score(y_val, y_val_pred_prob_ann)

print("Best ANN parameters:", best_params_ann)
print("Validation AUC (tuned ANN):", best_auc_ann)
print("Parameter comparison:")
print(pd.concat(
    [pd.DataFrame.from_dict(ann_default.get_params(), orient='index', columns=['Default']),
     pd.DataFrame.from_dict(ann_model_best.get_params(), orient='index', columns=['Tuned'])],
    axis=1))

# 7.8. Save ANN model
with open("model/ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)
