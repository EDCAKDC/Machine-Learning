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
################################################################################################
################################## Evaluate Model Performance on Validation Dataset ####################################
################################################################################################

## Run Start ##
# Load training data (for Logistic Regression model)
with open('file/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)

train_data = pd.read_csv("data/train_data_notscaled.csv",
                         encoding="GBK", index_col=0)
X_train_logist = train_data[significant_vars_multi]
X_train_logist_const = sm.add_constant(X_train_logist)
y_train_logist = train_data['Outcome']

# Load scaled training data (for ML models)
train_data_scaled = pd.read_csv(
    "data/train_data_scaled.csv", encoding="GBK", index_col=0)
X = train_data_scaled.loc[:, train_data_scaled.columns != 'Outcome']
y = train_data_scaled['Outcome']

# Split data into training and validation sets (same seed as model training)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  random_state=123, stratify=y)

###################### 1. Load Trained Models ##########################
with open("model/logistic_model.pkl", 'rb') as f:
    logist_model = pickle.load(f)

with open("model/tree_model.pkl", 'rb') as f:
    tree_model = pickle.load(f)

with open("model/rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)

with open("model/xgb_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)

with open("model/lgb_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)

with open("model/svm_model.pkl", 'rb') as f:
    svm_model = pickle.load(f)

with open("model/ann_model.pkl", 'rb') as f:
    ann_model = pickle.load(f)
## Run End ##

###################### 2. Get Predictions on Validation Set ##########################
# 2.1. Logistic model (evaluated on training set since no validation set is used)
y_train_pred_prob_logist = logist_model.predict(
    X_train_logist_const)  # predicted probabilities
y_train_pred_logist = (y_train_pred_prob_logist >= 0.5).astype(
    int)     # predicted classes

# 2.2. Decision Tree
y_val_pred_prob_tree = tree_model.predict_proba(X_val)[:, 1]
y_val_pred_tree = (y_val_pred_prob_tree >= 0.5).astype(int)

# 2.3. Random Forest
y_val_pred_prob_rf = rf_model.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_pred_prob_rf >= 0.5).astype(int)

# 2.4. XGBoost
y_val_pred_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_xgb = (y_val_pred_prob_xgb >= 0.5).astype(int)

# 2.5. LightGBM
y_val_pred_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
y_val_pred_lgb = (y_val_pred_prob_lgb >= 0.5).astype(int)

# 2.6. SVM
y_val_pred_prob_svm = svm_model.predict_proba(X_val)[:, 1]
y_val_pred_svm = (y_val_pred_prob_svm >= 0.5).astype(int)

# 2.7. ANN
y_val_pred_prob_ann = ann_model.predict_proba(X_val)[:, 1]
y_val_pred_ann = (y_val_pred_prob_ann >= 0.5).astype(int)

###################### Confusion Matrix Plot Function ##########################


def CM_plot(cm, model_name="Model"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_confusion_matrix.png", dpi=300)
    plt.close()
###################### Function to calculate metrics ##########################


def calculate_acc_pre_sen_f1_spc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp)
    return accuracy, precision, sensitivity, f1score, specificity


###################### Evaluate all models and collect results ##########################
model_results = []

models = {
    "Logistic": (y_train_logist, y_train_pred_logist),
    "DecisionTree": (y_val, y_val_pred_tree),
    "RandomForest": (y_val, y_val_pred_rf),
    "XGBoost": (y_val, y_val_pred_xgb),
    "LightGBM": (y_val, y_val_pred_lgb),
    "SVM": (y_val, y_val_pred_svm),
    "ANN": (y_val, y_val_pred_ann)
}

for model_name, (y_true, y_pred) in models.items():
    cm = confusion_matrix(y_true, y_pred)
    CM_plot(cm, model_name=model_name)
    acc, prec, sens, f1, spec = calculate_acc_pre_sen_f1_spc(cm)
    print(f"{model_name} → Accuracy: {acc:.3f}, Precision: {prec:.3f}, Sensitivity: {sens:.3f}, F1: {f1:.3f}, Specificity: {spec:.3f}")
    model_results.append([model_name, acc, prec, sens, f1, spec])

results_df = pd.DataFrame(model_results, columns=[
                          "Model", "Accuracy", "Precision", "Sensitivity", "F1 Score", "Specificity"])
results_df.to_csv("table/model_metrics_summary.csv",
                  index=False, encoding="utf-8-sig")

###################### AUC + Confidence Interval Calculation ##########################


def calculate_auc(y_label, y_pred_prob):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    se_auc = np.sqrt((auc_value * (1 - auc_value)) / len(y_label))
    z = norm.ppf(0.975)
    auc_ci_lower = auc_value - z * se_auc
    auc_ci_upper = auc_value + z * se_auc
    return auc_value, auc_ci_lower, auc_ci_upper

###################### ROC Curve Plotting ##########################


def ROC_plot(y_label, y_pred_prob, auc_value, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_ROC.png", dpi=300)
    plt.close()


###################### Calculate and Save AUC Results + ROC Plots ##########################
model_outputs = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist),
    "DecisionTree": (y_val, y_val_pred_prob_tree),
    "RandomForest": (y_val, y_val_pred_prob_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb),
    "SVM": (y_val, y_val_pred_prob_svm),
    "ANN": (y_val, y_val_pred_prob_ann)
}

auc_result_list = []

for model_name, (y_true, y_prob) in model_outputs.items():
    auc_val, auc_low, auc_up = calculate_auc(y_true, y_prob)
    ROC_plot(y_true, y_prob, auc_val, model_name)
    print(f"{model_name} AUC: {auc_val:.3f} (95% CI: {auc_low:.3f} - {auc_up:.3f})")
    auc_result_list.append([model_name, auc_val, auc_low, auc_up])

auc_df = pd.DataFrame(auc_result_list, columns=[
                      "Model", "AUC", "CI_Lower", "CI_Upper"])
auc_df.to_csv("table/model_auc_summary.csv", index=False, encoding="utf-8-sig")

###################### Calibration Curve Plotting ##########################


def CaliC_plot(y_label, y_pred_prob, model_name, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_label, y_pred_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--',
             color='gray', label='Perfect calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title(f"Calibration Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_calibration_curve.png", dpi=300)
    plt.close()

###################### Net Benefit Calculation Function ##########################


def calculate_net_benefi(y_label, y_pred_prob, thresholds=np.linspace(0.01, 1, 100)):
    net_benefit_model = []
    net_benefit_alltrt = []
    net_benefits_notrt = [0] * len(thresholds)
    total_obs = len(y_label)
    for thresh in thresholds:
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        nb_model = (tp / total_obs) - (fp / total_obs) * \
            (thresh / (1 - thresh))
        net_benefit_model.append(nb_model)
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total_right = tp + tn
        nb_all = (tp / total_right) - (tn / total_right) * \
            (thresh / (1 - thresh))
        net_benefit_alltrt.append(nb_all)
    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt

###################### DCA Plotting Function ##########################


def DCA_plot(net_benefit_model, net_benefit_alltrt, net_benefits_notrt,
             model_name, thresholds=np.linspace(0.01, 0.99, 100)):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model,
             label="Model Net Benefit", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_alltrt,
             label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefits_notrt, linestyle='--',
             color='green', label="Treat None", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(f"DCA Curve: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_dca_curve.png", dpi=300)
    plt.close()
###################### Function to calculate metrics ##########################


def calculate_acc_pre_sen_f1_spc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp)
    return accuracy, precision, sensitivity, f1score, specificity


###################### Evaluate all models and collect results ##########################
model_results = []

models = {
    "Logistic": (y_train_logist, y_train_pred_logist),
    "DecisionTree": (y_val, y_val_pred_tree),
    "RandomForest": (y_val, y_val_pred_rf),
    "XGBoost": (y_val, y_val_pred_xgb),
    "LightGBM": (y_val, y_val_pred_lgb),
    "SVM": (y_val, y_val_pred_svm),
    "ANN": (y_val, y_val_pred_ann)
}

for model_name, (y_true, y_pred) in models.items():
    cm = confusion_matrix(y_true, y_pred)
    CM_plot(cm, model_name=model_name)
    acc, prec, sens, f1, spec = calculate_acc_pre_sen_f1_spc(cm)
    print(f"{model_name} → Accuracy: {acc:.3f}, Precision: {prec:.3f}, Sensitivity: {sens:.3f}, F1: {f1:.3f}, Specificity: {spec:.3f}")
    model_results.append([model_name, acc, prec, sens, f1, spec])

results_df = pd.DataFrame(model_results, columns=[
                          "Model", "Accuracy", "Precision", "Sensitivity", "F1 Score", "Specificity"])
results_df.to_csv("table/model_metrics_summary.csv",
                  index=False, encoding="utf-8-sig")

###################### AUC + Confidence Interval Calculation ##########################


def calculate_auc(y_label, y_pred_prob):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    se_auc = np.sqrt((auc_value * (1 - auc_value)) / len(y_label))
    z = norm.ppf(0.975)
    auc_ci_lower = auc_value - z * se_auc
    auc_ci_upper = auc_value + z * se_auc
    return auc_value, auc_ci_lower, auc_ci_upper

###################### ROC Curve Plotting ##########################


def ROC_plot(y_label, y_pred_prob, auc_value, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_ROC.png", dpi=300)
    plt.close()


###################### Calculate and Save AUC Results + ROC Plots ##########################
model_outputs = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist),
    "DecisionTree": (y_val, y_val_pred_prob_tree),
    "RandomForest": (y_val, y_val_pred_prob_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb),
    "SVM": (y_val, y_val_pred_prob_svm),
    "ANN": (y_val, y_val_pred_prob_ann)
}

auc_result_list = []

for model_name, (y_true, y_prob) in model_outputs.items():
    auc_val, auc_low, auc_up = calculate_auc(y_true, y_prob)
    ROC_plot(y_true, y_prob, auc_val, model_name)
    print(f"{model_name} AUC: {auc_val:.3f} (95% CI: {auc_low:.3f} - {auc_up:.3f})")
    auc_result_list.append([model_name, auc_val, auc_low, auc_up])

auc_df = pd.DataFrame(auc_result_list, columns=[
                      "Model", "AUC", "CI_Lower", "CI_Upper"])
auc_df.to_csv("table/model_auc_summary.csv", index=False, encoding="utf-8-sig")

###################### Calibration Curve Plotting ##########################


def CaliC_plot(y_label, y_pred_prob, model_name, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_label, y_pred_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--',
             color='gray', label='Perfect calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title(f"Calibration Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_calibration_curve.png", dpi=300)
    plt.close()

###################### Net Benefit Calculation Function ##########################


def calculate_net_benefi(y_label, y_pred_prob, thresholds=np.linspace(0.01, 1, 100)):
    net_benefit_model = []
    net_benefit_alltrt = []
    net_benefits_notrt = [0] * len(thresholds)
    total_obs = len(y_label)
    for thresh in thresholds:
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        nb_model = (tp / total_obs) - (fp / total_obs) * \
            (thresh / (1 - thresh))
        net_benefit_model.append(nb_model)
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total_right = tp + tn
        nb_all = (tp / total_right) - (tn / total_right) * \
            (thresh / (1 - thresh))
        net_benefit_alltrt.append(nb_all)
    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt

###################### DCA Plotting Function ##########################


def DCA_plot(net_benefit_model, net_benefit_alltrt, net_benefits_notrt,
             model_name, thresholds=np.linspace(0.01, 0.99, 100)):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model,
             label="Model Net Benefit", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_alltrt,
             label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefits_notrt, linestyle='--',
             color='green', label="Treat None", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(f"DCA Curve: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot/{model_name}_dca_curve.png", dpi=300)
    plt.close()
###################### Model dictionary + batch execution ##########################


model_prob_outputs = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist),
    "DecisionTree": (y_val, y_val_pred_prob_tree),
    "RandomForest": (y_val, y_val_pred_prob_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb),
    "SVM": (y_val, y_val_pred_prob_svm),
    "ANN": (y_val, y_val_pred_prob_ann)
}

for model_name, (y_true, y_pred_prob) in model_prob_outputs.items():
    # Calibration plot
    CaliC_plot(y_true, y_pred_prob, model_name)

    # DCA calculation + plotting
    net_model, net_all, net_none = calculate_net_benefi(y_true, y_pred_prob)
    DCA_plot(net_model, net_all, net_none, model_name)

###################### Metrics summary for all models on training/validation sets ##########################
models = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist, y_train_pred_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree, y_val_pred_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf, y_val_pred_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb, y_val_pred_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb, y_val_pred_lgb),
    "SVM": (y_val, y_val_pred_prob_svm, y_val_pred_svm),
    "ANN": (y_val, y_val_pred_prob_ann, y_val_pred_ann)
}

# Storage for evaluation results
results = []

# Unified metric evaluation function


def calculate_metrics(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    auc = roc_auc_score(y_true, y_pred_prob)
    se = np.sqrt((auc * (1 - auc)) / len(y_true))
    z = norm.ppf(0.975)
    auc_lower = auc - z * se
    auc_upper = auc + z * se
    return auc, auc_lower, auc_upper, accuracy, precision, sensitivity, specificity, f1


####################### ROC Curves #########################
plt.figure(figsize=(8, 6))
for name, (y_true, y_prob, y_pred) in models.items():
    auc, auc_lower, auc_upper, acc, pre, sen, spec, f1 = calculate_metrics(
        y_true, y_pred, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    results.append([name, auc, auc_lower, auc_upper, acc, pre, sen, spec, f1])
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("prediction/ROC_curves_allmodel_validation.png", dpi=300)
plt.close()

####################### Calibration Curves #########################
plt.figure(figsize=(10, 8))
for name, (y_true, y_prob, _) in models.items():
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)
plt.plot([0, 1], [0, 1], linestyle='--',
         color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models")
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig("prediction/Calibration_curves_allmodel_validation.png", dpi=300)
plt.close()

####################### DCA Curves #########################


def calculate_net_benefi(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 100)):
    nb_model, nb_all, nb_none = [], [], [0]*len(thresholds)
    total = len(y_true)
    for thresh in thresholds:
        pred = y_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        net_model = (tp / total) - (fp / total) * (thresh / (1 - thresh))
        nb_model.append(net_model)
        tn, fp, fn, tp = confusion_matrix(y_true, y_true).ravel()
        total_right = tp + tn
        net_all = (tp / total_right) - (tn / total_right) * \
            (thresh / (1 - thresh))
        nb_all.append(net_all)
    return nb_model, nb_all, nb_none


plt.figure(figsize=(10, 8))
for name, (y_true, y_prob, _) in models.items():
    nb_model, nb_all, nb_none = calculate_net_benefi(y_true, y_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), nb_model, label=name)
plt.plot(np.linspace(0.01, 0.99, 100), nb_all,
         '--', color='red', label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), nb_none,
         '--', color='green', label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("prediction/DCA_curves_allmodel_validation.png", dpi=300)
plt.close()
####################### Save summarized results #########################
results_df = pd.DataFrame(results, columns=[
    "Model", "AUC", "AUC 95% CI Lower", "AUC 95% CI Upper", "Accuracy",
    "Precision", "Sensitivity", "Specificity", "F1 Score"
])
results_df.to_csv("prediction/model_performance_validation.csv",
                  index=False, encoding="utf-8-sig")

################################################################################################
################################## Model evaluation on test dataset ####################################
################################################################################################
## Run start ##
# Load test dataset (for Logistic model)
with open('file/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)

significant_vars_multi

test_data = pd.read_csv("data/test_data_notscaled.csv",
                        encoding="GBK", index_col=0)
X_test_logist = test_data[significant_vars_multi]
X_test_logist_const = sm.add_constant(X_test_logist)
y_test_logist = test_data['Outcome']

# Load test dataset (for ML models)
test_data_scaled = pd.read_csv(
    "data/test_data_scaled.csv", encoding="GBK", index_col=0)
X_test = test_data_scaled.loc[:, test_data_scaled.columns != 'Outcome']
y_test = test_data_scaled['Outcome']
## Run end ##

###################### 1. Predictions on test dataset ##########################
# 1.1. Logistic model
y_test_pred_prob_logist = logist_model.predict(X_test_logist_const)
y_test_pred_logist = (y_test_pred_prob_logist >= 0.5).astype(int)

# 1.2. Decision Tree
y_test_pred_prob_tree = tree_model.predict_proba(X_test)[:, 1]
y_test_pred_tree = (y_test_pred_prob_tree >= 0.5).astype(int)

# 1.3. Random Forest
y_test_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_pred_prob_rf >= 0.5).astype(int)

# 1.4. XGBoost
y_test_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_pred_prob_xgb >= 0.5).astype(int)

# 1.5. LightGBM
y_test_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_test_pred_lgb = (y_test_pred_prob_lgb >= 0.5).astype(int)

# 1.6. SVM
y_test_pred_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_test_pred_svm = (y_test_pred_prob_svm >= 0.5).astype(int)

# 1.7. ANN
y_test_pred_prob_ann = ann_model.predict_proba(X_test)[:, 1]
y_test_pred_ann = (y_test_pred_prob_ann >= 0.5).astype(int)

###################### 2. Confusion Matrix calculation and visualization ##########################
# 2.1. Logistic model
cm_logist_test = confusion_matrix(y_test_logist, y_test_pred_logist)
CM_plot(cm_logist_test)

# 2.2. Decision Tree
cm_tree_test = confusion_matrix(y_test, y_test_pred_tree)
# CM_plot(cm_tree_test)

# 2.3. Random Forest
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)
# CM_plot(cm_rf_test)

# 2.4. XGBoost
cm_xgb_test = confusion_matrix(y_test, y_test_pred_xgb)
# CM_plot(cm_xgb_test)

# 2.5. LightGBM
cm_lgb_test = confusion_matrix(y_test, y_test_pred_lgb)
# CM_plot(cm_lgb_test)

# 2.6. SVM
cm_svm_test = confusion_matrix(y_test, y_test_pred_svm)
# CM_plot(cm_svm_test)

# 2.7. ANN
cm_ann_test = confusion_matrix(y_test, y_test_pred_ann)
CM_plot(cm_ann_test)
###################### Calculate evaluation metrics: accuracy, precision, sensitivity, F1 score, specificity ##########################
# Logistic model
accuracy_logist_test, precision_logist_test, sensitivity_logist_test, f1_logist_test, specificity_logist_test = calculate_acc_pre_sen_f1_spc(
    cm_logist_test)
# Decision Tree
accuracy_tree_test, precision_tree_test, sensitivity_tree_test, f1_tree_test, specificity_tree_test = calculate_acc_pre_sen_f1_spc(
    cm_tree_test)
# Random Forest
accuracy_rf_test, precision_rf_test, sensitivity_rf_test, f1_rf_test, specificity_rf_test = calculate_acc_pre_sen_f1_spc(
    cm_rf_test)
# XGBoost
accuracy_xgb_test, precision_xgb_test, sensitivity_xgb_test, f1_xgb_test, specificity_xgb_test = calculate_acc_pre_sen_f1_spc(
    cm_xgb_test)
# LightGBM
accuracy_lgb_test, precision_lgb_test, sensitivity_lgb_test, f1_lgb_test, specificity_lgb_test = calculate_acc_pre_sen_f1_spc(
    cm_lgb_test)
# SVM
accuracy_svm_test, precision_svm_test, sensitivity_svm_test, f1_svm_test, specificity_svm_test = calculate_acc_pre_sen_f1_spc(
    cm_svm_test)
# ANN
accuracy_ann_test, precision_ann_test, sensitivity_ann_test, f1_ann_test, specificity_ann_test = calculate_acc_pre_sen_f1_spc(
    cm_ann_test)

###################### Calculate AUC and 95% Confidence Interval ##########################
# Logistic model
auc_value_logist_test, auc_ci_lower_logist_test, auc_ci_upper_logist_test = calculate_auc(
    y_test_logist, y_test_pred_prob_logist)
# Decision Tree
auc_value_tree_test, auc_ci_lower_tree_test, auc_ci_upper_tree_test = calculate_auc(
    y_test, y_test_pred_prob_tree)
# Random Forest
auc_value_rf_test, auc_ci_lower_rf_test, auc_ci_upper_rf_test = calculate_auc(
    y_test, y_test_pred_prob_rf)
# XGBoost
auc_value_xgb_test, auc_ci_lower_xgb_test, auc_ci_upper_xgb_test = calculate_auc(
    y_test, y_test_pred_prob_xgb)
# LightGBM
auc_value_lgb_test, auc_ci_lower_lgb_test, auc_ci_upper_lgb_test = calculate_auc(
    y_test, y_test_pred_prob_lgb)
# SVM
auc_value_svm_test, auc_ci_lower_svm_test, auc_ci_upper_svm_test = calculate_auc(
    y_test, y_test_pred_prob_svm)
# ANN
auc_value_ann_test, auc_ci_lower_ann_test, auc_ci_upper_ann_test = calculate_auc(
    y_test, y_test_pred_prob_ann)

###################### Summary of model performance on the test set ##########################
model_results_test = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "SVM", "ANN"],
    "AUC": [auc_value_logist_test, auc_value_tree_test, auc_value_rf_test, auc_value_xgb_test, auc_value_lgb_test, auc_value_svm_test, auc_value_ann_test],
    "AUC 95% CI Lower": [auc_ci_lower_logist_test, auc_ci_lower_tree_test, auc_ci_lower_rf_test, auc_ci_lower_xgb_test, auc_ci_lower_lgb_test, auc_ci_lower_svm_test, auc_ci_lower_ann_test],
    "AUC 95% CI Upper": [auc_ci_upper_logist_test, auc_ci_upper_tree_test, auc_ci_upper_rf_test, auc_ci_upper_xgb_test, auc_ci_upper_lgb_test, auc_ci_upper_svm_test, auc_ci_upper_ann_test],
    "Accuracy": [accuracy_logist_test, accuracy_tree_test, accuracy_rf_test, accuracy_xgb_test, accuracy_lgb_test, accuracy_svm_test, accuracy_ann_test],
    "Precision": [precision_logist_test, precision_tree_test, precision_rf_test, precision_xgb_test, precision_lgb_test, precision_svm_test, precision_ann_test],
    "Sensitivity": [sensitivity_logist_test, sensitivity_tree_test, sensitivity_rf_test, sensitivity_xgb_test, sensitivity_lgb_test, sensitivity_svm_test, sensitivity_ann_test],
    "Specificity": [specificity_logist_test, specificity_tree_test, specificity_rf_test, specificity_xgb_test, specificity_lgb_test, specificity_svm_test, specificity_ann_test],
    "F1 Score": [f1_logist_test, f1_tree_test, f1_rf_test, f1_xgb_test, f1_lgb_test, f1_svm_test, f1_ann_test]
})
model_results_test.to_csv("prediction/model_performance_test.csv", index=False)

###################### Plot ROC Curves ##########################
plt.figure(figsize=(8, 6))
models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist, auc_value_logist_test),
    "Decision Tree": (y_test, y_test_pred_prob_tree, auc_value_tree_test),
    "Random Forest": (y_test, y_test_pred_prob_rf, auc_value_rf_test),
    "XGBoost": (y_test, y_test_pred_prob_xgb, auc_value_xgb_test),
    "LightGBM": (y_test, y_test_pred_prob_lgb, auc_value_lgb_test),
    "SVM": (y_test, y_test_pred_prob_svm, auc_value_svm_test),
    "ANN": (y_test, y_test_pred_prob_ann, auc_value_ann_test)
}
for model_name, (y_true, y_pred_prob, auc_value) in models_test.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models (Test Set)")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("prediction/ROC_curves_allmodel_test.png", dpi=300)
plt.show()

###################### Plot Calibration Curves ##########################
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
plt.plot([0, 1], [0, 1], linestyle='--',
         color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models (Test Set)")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("prediction/Calibration_curves_allmodel_test.png", dpi=300)
plt.show()
###################### Plot Decision Curve Analysis (DCA) ##########################
plt.figure(figsize=(10, 8))

# Plot Net Benefit for each model
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(
        y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name)

# Plot Treat All and Treat None strategies
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt,
         linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt,
         linestyle="--", color="green", label="Treat None")

plt.xlabel("Threshold Probability")
plt.ylim(-0.2, np.nanmax(np.array(net_benefit)) + 0.1)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models (Test Set)")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("prediction/DCA_curves_allmodel_test.png", dpi=300)
plt.show()
###################### Plot SHAP Interpretations ##########################

# Create output directories
PLOT_DIR = "shap/plots/"
HTML_DIR = "shap/html/"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# Define SHAP explanation function


def explain_once(model, X_train, X_test,
                 model_name,
                 n_interpret=50,
                 obs_index=0,
                 scatter_x=None,
                 scatter_color=None,
                 is_classifier=True,
                 explainer_type="kernel"):
    """Generate and save SHAP interpretation plots."""
    # Sample background and evaluation set
    X_bg = shap.sample(X_train, 100, random_state=0)
    X_eval = X_test.iloc[:n_interpret, :]

    # Define prediction function
    if is_classifier:
        f = (lambda x: model.predict_proba(x)
             if hasattr(model, "predict_proba")
             else model.predict(x))
    else:
        f = model.predict

    # Build SHAP explainer
    explainer = (shap.TreeExplainer(model, X_bg)
                 if explainer_type == "tree"
                 else shap.KernelExplainer(f, X_bg))

    shap_values_full = explainer.shap_values(X_eval)

    # Parse SHAP values and base value
    if is_classifier:
        shap_vals = shap_values_full[1] if isinstance(
            shap_values_full, list) else shap_values_full
        base_val = (explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value)
    else:
        shap_vals = shap_values_full
        base_val = explainer.expected_value

    cols = X_eval.columns.tolist()
    col_x = scatter_x or cols[0]
    col_c = scatter_color or (cols[1] if len(cols) > 1 else cols[0])

    # Summary bar plot
    shap.summary_plot(shap_vals, X_eval, plot_type="bar",
                      max_display=15, show=False)
    plt.savefig(f"{PLOT_DIR}bar_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Summary beeswarm plot
    shap.summary_plot(shap_vals, X_eval, max_display=15, show=False)
    plt.savefig(f"{PLOT_DIR}beeswarm_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Convert to Explanation object if needed
    if not isinstance(shap_vals, shap._explanation.Explanation):
        shap_exp = shap.Explanation(
            values=shap_vals,
            base_values=np.repeat(base_val, shap_vals.shape[0]),
            data=X_eval.values,
            feature_names=cols
        )
    else:
        shap_exp = shap_vals

    # Scatter plot
    shap.plots.scatter(shap_exp[:, col_x],
                       color=shap_exp[:, col_c], show=False)
    plt.title(f"SHAP scatter: {col_x} vs color={col_c}")
    plt.savefig(f"{PLOT_DIR}scatter_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Waterfall plot for one observation
    shap.plots.waterfall(
        shap.Explanation(values=shap_exp[obs_index].values,
                         base_values=base_val,
                         data=X_eval.iloc[obs_index].values,
                         feature_names=cols),
        max_display=15, show=False)
    plt.savefig(f"{PLOT_DIR}waterfall_{obs_index}_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Force plots (single and batch)
    html_one = shap.plots.force(
        base_val, shap_exp[obs_index], X_eval.iloc[obs_index])
    shap.save_html(f"{HTML_DIR}force_single_{model_name}.html", html_one)

    html_all = shap.plots.force(base_val, shap_exp, X_eval)
    shap.save_html(f"{HTML_DIR}force_all_{model_name}.html", html_all)

    print(f"✅ {model_name}: SHAP plots saved")


# Define model list for interpretation
models_info = [
    dict(name="logist", model=logist_model,
         X_train=X_train_logist_const, X_test=X_test_logist_const,
         is_classifier=True, scatter_x="Pregnancies", scatter_color="Glucose"),

    dict(name="tree",   model=tree_model,
         X_train=X_train, X_test=X_test,
         is_classifier=True),

    dict(name="rf",     model=rf_model,
         X_train=X_train, X_test=X_test,
         is_classifier=True, explainer_type="tree"),

    dict(name="xgb",    model=xgb_model,
         X_train=X_train, X_test=X_test,
         is_classifier=True, explainer_type="tree"),

    dict(name="lgb",    model=lgb_model,
         X_train=X_train, X_test=X_test,
         is_classifier=True, explainer_type="tree"),

    dict(name="svm",    model=svm_model,
         X_train=X_train, X_test=X_test,
         is_classifier=True),

    dict(name="ann",    model=ann_model,
         X_train=X_train, X_test=X_test,
         is_classifier=True)
]

# Batch run SHAP explanation for all models
for info in models_info:
    explain_once(model=info["model"],
                 X_train=info["X_train"],
                 X_test=info["X_test"],
                 model_name=info["name"],
                 n_interpret=50,
                 obs_index=5,
                 scatter_x=info.get("scatter_x"),
                 scatter_color=info.get("scatter_color"),
                 is_classifier=info["is_classifier"],
                 explainer_type=info.get("explainer_type", "kernel"))
