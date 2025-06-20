# Machine-Learning
# 🔍 Binary Classification Models with Randomized Search

👋 Hi there! I'm a beginner in machine learning, and this is a personal project where I explore various **binary classification models**.

 At first, I used **grid search** to tune hyperparameters, but I quickly realized it's too slow when the parameter space grows.  
 So, to save time and speed things up, I switched to **RandomizedSearchCV** for hyperparameter optimization. It’s much faster and still performs well!

📦 Models included in this repo:
- 🌳 Decision Tree
- 🌲 Random Forest
- 🚀 XGBoost
- 💡 LightGBM
- 📐 Support Vector Machine (SVM)
- 🧠 Artificial Neural Network (ANN)

✅ Each model is evaluated using **AUC on a validation set**, and the best models are saved for reuse.

---

## 🔮 Future Plans
-  Model interpretability (e.g. SHAP, permutation importance)
-  Survival models (e.g. Cox, RSF)
-  Try AutoML tools
-  Visualization and report generation

---

📌 This repository is a learning journey and will be updated as I grow.  
Feel free to fork, use, or suggest improvements! 
 📅 **Last updated: June 16, 2025**  
> 🚀 **Update Summary:**  
> - Preprocessed and cleaned the dataset (handling zeros in continuous variables)  
> - Visualized variable distributions before and after correction  
> - Performed feature selection using univariate/multivariate logistic regression and LASSO  
> - Trained and tuned the following models:  
>   - Logistic Regression  
>   - Decision Tree  
>   - Random Forest  
>   - XGBoost  
>   - LightGBM  
>   - Support Vector Machine (SVM)  
>   - Artificial Neural Network (ANN)  
> - Saved all trained models and significant variables  
> - Visualized tree structure and histograms  

---

## 📊 Dataset

- **Source:** [Kaggle - Diabetes Data Set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
- **Samples:** 768
- **Target Variable:** `Outcome` (1 = diabetic, 0 = non-diabetic)
- **Features:**  
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
- **Note:** Zero values in some features were treated as missing and imputed with medians.

---

## 📂 Project Structure

```bash
.
├── data/                      # Cleaned and processed datasets
│   ├── diabetes.csv
│   ├── data_imputed.csv
│   ├── train_data_notscaled.csv
│   ├── train_data_scaled.csv
│   ├── test_data_notscaled.csv
│   └── test_data_scaled.csv
│
├── file/                      # Saved feature list from logistic regression
│   └── significant_vars.pkl
│
├── model/                     # Trained models (pickle format)
│   ├── logistic_model.pkl
│   ├── tree_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── svm_model.pkl
│   └── ann_model.pkl
│
├── plot/                      # Visualizations
│   ├── Age_hist.png
│   ├── BloodPressure_hist.png
│   ├── BloodPressure_fixed.png
│   ├── BMI_hist.png
│   ├── BMI_fixed.png
│   ├── DiabetesPedigreeFunction_hist.png
│   ├── Glucose_hist.png
│   ├── Glucose_fixed.png
│   ├── Insulin_hist.png
│   ├── Insulin_fixed.png
│   ├── SkinThickness_hist.png
│   ├── SkinThickness_fixed.png
│   └── tree_structure.jpg

📌 Update on June 20, 2025
Added full SHAP-based model interpretation pipeline (model_shap_interpretation.py), covering 7 classification models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, SVM, and ANN.

Automatically generates SHAP bar plots and beeswarm plots for feature importance visualization.

For each model, the script samples 50 test samples and uses either TreeExplainer or KernelExplainer based on model type.

SHAP scatter plots, waterfall plots, and force plots are structured in the code, but currently only bar and beeswarm plots are generated successfully.

This module will be further refined as actual debugging issues arise.

