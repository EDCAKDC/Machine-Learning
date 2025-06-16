############################### 2. Decision Tree Model (Optimized with RandomizedSearchCV) ###############################
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import scipy.stats as stats

# 2.1 Load the standardized training dataset
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="GBK", index_col=0)

# 2.2 Separate features and target variable
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes']

# 2.3 Split the dataset into training and validation sets (70% training, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y)

# 2.4 Train a CART decision tree with default parameters
tree_default = DecisionTreeClassifier(random_state=123)
tree_default.fit(X_train, y_train)
print("Default model parameters:", pd.DataFrame.from_dict(
    tree_default.get_params(), orient='index'))

# 2.5 Evaluate the default model's AUC on the validation set
y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1]
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed)
print("AUC of the default model on the validation set:", auc_treed)

# 2.6 Define the hyperparameter search space (customizable search count)
param_dist = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [5, 10, 20],
    'max_features': ['sqrt', None],
    # Continuous uniform distribution for pruning penalty
    'ccp_alpha': stats.uniform(0.0, 0.1)
}

# 2.7 Define the base model and AUC scoring function
tree_model = DecisionTreeClassifier(random_state=123)
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 2.8 Perform randomized search over 30 combinations
random_search = RandomizedSearchCV(
    estimator=tree_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=auc_scorer,
    cv=5,
    random_state=123,
    n_jobs=-1,
    verbose=1
)

# 2.9 Fit the randomized search on the training set
random_search.fit(X_train, y_train)

# 2.10 Retrieve the best model and its hyperparameters
tree_model_best = random_search.best_estimator_
best_params = random_search.best_params_
print("Best hyperparameter combination:", best_params)

# 2.11 Evaluate the best model's AUC on the validation set
y_val_pred_prob_best = tree_model_best.predict_proba(X_val)[:, 1]
best_auc_tree = roc_auc_score(y_val, y_val_pred_prob_best)
print("AUC of the best model on the validation set:", best_auc_tree)


################### 3. Random Forest (RF) Model ##########################

# 3.1 Load standardized training data
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="utf-8", index_col=0)

# 3.2 Separate features and target
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes']

# 3.3 Split into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y)

# 3.4 Train Random Forest model with default parameters
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True)
rf_model_default.fit(X_train, y_train)

# Show default parameters
print("Default RF model parameters:", pd.DataFrame.from_dict(
    rf_model_default.get_params(), orient='index'))

# 3.5 Evaluate default model on validation set
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("AUC of the default RF model on the validation set:", auc_rfd)

# 3.6 Define hyperparameter search space
param_dist = {
    'n_estimators': np.arange(50, 501, 50),  # Number of trees: 50 to 500
    # Number of features to consider at each split
    'max_features': list(range(2, round(np.sqrt(X.shape[1])) + 1))
}

# 3.7 Define scorer using AUC
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 3.8 Perform randomized search with 30 iterations
rf_model = RandomForestClassifier(random_state=123)
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=auc_scorer,
    cv=5,
    random_state=123,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)

# 3.9 Get best model and parameters
rf_model_best = random_search.best_estimator_
best_params_rf = random_search.best_params_
best_auc_rf = random_search.best_score_

# 3.10 Evaluate best model on validation set
y_val_pred_prob_best = rf_model_best.predict_proba(X_val)[:, 1]
val_auc_best = roc_auc_score(y_val, y_val_pred_prob_best)

# 3.11 Output results
print("Best RF parameter combination:", best_params_rf)
print("Validation AUC of the default RF model:", auc_rfd)
print("Validation AUC of the tuned RF model:", val_auc_best)
print("Best RF model parameters:", pd.DataFrame.from_dict(
    rf_model_best.get_params(), orient='index'))

# 3.12 Save the best model
with open("trained_models/rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)

###################### 4. XGBoost Model ##########################

# 4.1 Load standardized training data
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="utf-8", index_col=0)

# 4.2 Separate features and target
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes']

# 4.3 Split into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y)

# 4.4 Train XGBoost with default parameters
xgb_default = XGBClassifier(
    random_state=123, use_label_encoder=False, eval_metric='logloss')
xgb_default.fit(X_train, y_train)

# Evaluate default model on validation set
y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("AUC of the default XGBoost model on the validation set:", auc_xgbd)

# 4.5 Define hyperparameter search space
param_dist = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0]
}

# 4.6 Define AUC scoring function
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 4.7 Perform randomized search with 30 iterations
xgb_model = XGBClassifier(
    random_state=123, use_label_encoder=False, eval_metric='logloss')

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=auc_scorer,
    cv=5,
    random_state=123,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)

# 4.8 Retrieve best model and parameters
xgb_model_best = random_search.best_estimator_
best_params_xgb = random_search.best_params_
best_auc_xgb = random_search.best_score_

# 4.9 Evaluate best model on validation set
y_val_pred_prob_best = xgb_model_best.predict_proba(X_val)[:, 1]
val_auc_best = roc_auc_score(y_val, y_val_pred_prob_best)

# 4.10 Print results
print("Best XGBoost hyperparameter combination:", best_params_xgb)
print("Validation AUC of the default XGBoost model:", auc_xgbd)
print("Validation AUC of the tuned XGBoost model:", val_auc_best)
print("Best XGBoost model parameters:", pd.DataFrame.from_dict(
    xgb_model_best.get_params(), orient='index'))

# 4.11 Save the best model
with open("trained_models/xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)

###################### 5. LightGBM Model ##########################

# 5.1 Load standardized training data
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="utf-8", index_col=0)

# 5.2 Separate features and target
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes']

# 5.3 Split into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y)

# 5.4 Train LightGBM with default parameters
lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)

# Evaluate default model on validation set
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("AUC of the default LightGBM model on the validation set:", auc_lgbd)

# 5.5 Define hyperparameter search space
param_dist = {
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# 5.6 Define AUC scorer
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 5.7 Perform randomized search with 30 iterations
lgb_model = lgb.LGBMClassifier(random_state=123)
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=auc_scorer,
    cv=5,
    random_state=123,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)

# 5.8 Retrieve best model and parameters
lgb_model_best = random_search.best_estimator_
best_params_lgb = random_search.best_params_
best_auc_lgb = random_search.best_score_

# 5.9 Evaluate best model on validation set
y_val_pred_prob_best = lgb_model_best.predict_proba(X_val)[:, 1]
val_auc_best = roc_auc_score(y_val, y_val_pred_prob_best)

# 5.10 Print results
print("Best LightGBM hyperparameter combination:", best_params_lgb)
print("Validation AUC of the default LightGBM model:", auc_lgbd)
print("Validation AUC of the tuned LightGBM model:", val_auc_best)
print("Best LightGBM model parameters:", pd.DataFrame.from_dict(
    lgb_model_best.get_params(), orient='index'))

# 5.11 Save the best model
with open("trained_models/lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)

###################### 6. Support Vector Machine (SVM) Model ##########################

# 6.1 Load standardized training data
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="utf-8", index_col=0)

# 6.2 Separate features and target
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes']

# 6.3 Split into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y)

# 6.4 Train SVM with default parameters
svm_default = SVC(probability=True, random_state=123)
svm_default.fit(X_train, y_train)

# Evaluate default model on validation set
y_val_pred_prob_svmd = svm_default.predict_proba(X_val)[:, 1]
auc_svmd = roc_auc_score(y_val, y_val_pred_prob_svmd)
print("AUC of the default SVM model on the validation set:", auc_svmd)

# 6.5 Define custom parameter search space
param_dist = [
    {'kernel': ['linear'], 'C': [0.1, 2, 10]},
    {'kernel': ['rbf'], 'C': [0.1, 2, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]},
    {'kernel': ['poly'], 'C': [0.1, 2, 10], 'gamma': [
        'scale', 'auto', 0.01, 0.1, 1], 'degree': [2, 3, 4]}
]

# 6.6 Define AUC scorer
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 6.7 Perform randomized search
svm_model = SVC(probability=True, random_state=123)
random_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=auc_scorer,
    cv=5,
    random_state=123,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)

# 6.8 Retrieve best model and parameters
svm_model_best = random_search.best_estimator_
best_params_svm = random_search.best_params_
best_auc_svm = random_search.best_score_

# 6.9 Evaluate best model on validation set
y_val_pred_prob_best = svm_model_best.predict_proba(X_val)[:, 1]
val_auc_best = roc_auc_score(y_val, y_val_pred_prob_best)

# 6.10 Print results
print("Best SVM hyperparameter combination:", best_params_svm)
print("Validation AUC of the default SVM model:", auc_svmd)
print("Validation AUC of the tuned SVM model:", val_auc_best)
print("Parameters of default and tuned SVM models:")
print(pd.concat([
    pd.DataFrame.from_dict(svm_default.get_params(),
                           orient='index', columns=['Default']),
    pd.DataFrame.from_dict(svm_model_best.get_params(),
                           orient='index', columns=['Tuned'])
], axis=1))

# 6.11 Save the best model
with open("trained_models/svm_model.pkl", 'wb') as f:
    pickle.dump(svm_model_best, f)

###################### 7. Artificial Neural Network (ANN) Model ##########################

# 7.1 Load standardized training data
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="utf-8", index_col=0)

# 7.2 Separate features and target
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes']

# 7.3 Split into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y)

# 7.4 Train ANN with default parameters
ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)

# Evaluate default model on validation set
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("AUC of the default ANN model on the validation set:", auc_annd)

# 7.5 Define hyperparameter search space
param_dist = {
    'hidden_layer_sizes': [(25,), (50,), (100,), (10, 10), (50, 50), (100, 100), (50, 50, 50)],
    'activation': ['relu', 'tanh', 'logistic']
}

# 7.6 Define AUC scorer
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 7.7 Perform randomized search
ann_model = MLPClassifier(random_state=123, max_iter=500)
random_search = RandomizedSearchCV(
    estimator=ann_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=auc_scorer,
    cv=5,
    random_state=123,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)

# 7.8 Retrieve best model and parameters
ann_model_best = random_search.best_estimator_
best_params_ann = random_search.best_params_
best_auc_ann = random_search.best_score_

# 7.9 Evaluate best model on validation set
y_val_pred_prob_best = ann_model_best.predict_proba(X_val)[:, 1]
val_auc_best = roc_auc_score(y_val, y_val_pred_prob_best)

# 7.10 Print results
print("Best ANN hyperparameter combination:", best_params_ann)
print("Validation AUC of the default ANN model:", auc_annd)
print("Validation AUC of the tuned ANN model:", val_auc_best)
print("Parameters of default and tuned ANN models:")
print(pd.concat([
    pd.DataFrame.from_dict(ann_default.get_params(),
                           orient='index', columns=['Default']),
    pd.DataFrame.from_dict(ann_model_best.get_params(),
                           orient='index', columns=['Tuned'])
], axis=1))

# 7.11 Save the best model
with open("trained_models/ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)
