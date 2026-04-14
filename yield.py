# =========================================
# 1. IMPORT LIBRARY
# =========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# =========================================
# 2. DATA UNDERSTANDING
# =========================================
df = pd.read_csv("crop_yield.csv")

print("===== DATA AWAL =====")
print(df.head())

print("\n===== INFO DATA =====")
print(df.info())

print("\n===== STATISTIK DESKRIPTIF =====")
print(df.describe())

# =========================================
# 3. DATA PREPROCESSING
# =========================================

# Missing value
print("\nMissing Value:", df.isnull().sum().sum())
df = df.dropna()

# Duplikat
print("Data Duplikat:", df.duplicated().sum())
df = df.drop_duplicates()

# Encoding
cat_cols = ['crop', 'season', 'state']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# outlier (IQR)
Q1 = df['yield'].quantile(0.25)
Q3 = df['yield'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['yield'] >= Q1 - 1.5 * IQR) & 
        (df['yield'] <= Q3 + 1.5 * IQR)]

# Feature & target
X = df.drop(columns=['yield'])
y = df['yield']


# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================================
# 4. EDA (VISUALISASI)
# =========================================

# Distribusi target
plt.figure()
sns.histplot(y, kde=True)
plt.title("Distribusi Yield")
plt.show()

# Korelasi
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi")
plt.show()

# =========================================
# 5. DATA SPLITTING (3 VARIASI)
# =========================================

splits = {
    "70:30": 0.3,
    "80:20": 0.2,
    "90:10": 0.1
}

final_results = []

# =========================================
# 6. LOOP UNTUK SETIAP SPLIT
# =========================================

for split_name, test_size in splits.items():
    print(f"\n===== SPLIT {split_name} =====")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # =========================================
    # 7. HYPERPARAMETER TUNING
    # =========================================

    # Random Forest GridSearch
    rf_params = {
    'n_estimators': [100],
    'max_depth': [9],
    'min_samples_split': [5],
    'min_samples_leaf': [2]
    }

    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                           rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    # Gradient Boosting GridSearch
    gbr_params = {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }

    gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state=42),
                            gbr_params, cv=3, n_jobs=-1)
    gbr_grid.fit(X_train, y_train)
    gbr_best = gbr_grid.best_estimator_

    # XGBoost GridSearch
    xgb_params = {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }

    xgb_grid = GridSearchCV(XGBRegressor(random_state=42),
                            xgb_params, cv=3, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_

    # =========================================
    # 8. EVALUASI
    # =========================================

    def evaluate(model, name):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        final_results.append([split_name, name, mae, mse, rmse, r2])

    evaluate(rf_best, "Random Forest")
    evaluate(gbr_best, "Gradient Boosting")
    evaluate(xgb_best, "XGBoost")

# =========================================
# 9. HASIL AKHIR
# =========================================
results_df = pd.DataFrame(final_results, columns=[
    "Split", "Model", "MAE", "MSE", "RMSE", "R2 Score"
])

print("\n===== HASIL AKHIR SEMUA MODEL =====")
print(results_df.sort_values(by="R2 Score", ascending=False))

# =========================================
# 10. VISUALISASI PERBANDINGAN
# =========================================
plt.figure(figsize=(10,6))
sns.barplot(data=results_df, x="Model", y="R2 Score", hue="Split")
plt.title("Perbandingan R2 Score Model")
plt.show()