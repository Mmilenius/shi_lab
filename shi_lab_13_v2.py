import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import optuna
import warnings
warnings.filterwarnings("ignore")

# --- 1. Генерація синтетичних даних ---
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 10*X[:,0] + 5*X[:,1]**2 + np.random.normal(0, 0.3, 1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Базова модель ---
base_model = RandomForestRegressor(random_state=42)
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)
base_mae = mean_absolute_error(y_test, y_pred)
base_r2 = r2_score(y_test, y_pred)
print(f"Базова модель: MAE={base_mae:.3f}, R2={base_r2:.3f}")

# --- 3. Grid Search ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_best = grid_search.best_estimator_

# --- 4. Random Search ---
param_dist = {
    'n_estimators': np.arange(50, 250, 50),
    'max_depth': np.arange(3, 10),
    'min_samples_split': np.arange(2, 10)
}
random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                                   param_distributions=param_dist, n_iter=10, cv=3,
                                   scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
random_best = random_search.best_estimator_

# --- 5. Bayesian Optimization (Optuna) ---
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 250)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, show_progress_bar=False)
optuna_best_params = study.best_params
optuna_best = RandomForestRegressor(**optuna_best_params, random_state=42)
optuna_best.fit(X_train, y_train)

# --- 6. Оцінювання ---
models = {
    "GridSearch": grid_best,
    "RandomSearch": random_best,
    "BayesianOpt": optuna_best
}

print("\nРезультати оптимізації:")
for name, model in models.items():
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:12s} -> MAE={mae:.3f}, R2={r2:.3f}")

print("\nНайкращі параметри Optuna:", optuna_best_params)
