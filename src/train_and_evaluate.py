from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd


class TrainAndEvaluate:
    """
    Class for training and evaluating machine learning models on a given dataset.

    Attributes:
        df (pd.DataFrame): Input DataFrame for training and evaluation.
    """

    def __init__(self, df):
        """
        Initializes the TrainAndEvaluate class with a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def train_and_evaluate(self, target_column, test_size=0.25, models=None, problem_type='classification',
                        hyperparameter_tuning=False, hyperparameter_tuning_threshold=0.5, n_jobs=-1,
                        models_not_needed=None, performance_metric='accuracy'):
        """
        Trains and evaluates a list of machine learning models.

        Returns:
            tuple: Best model, best score, best parameters, all trained models with scores.
        """
        model_dict = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GaussianNB': GaussianNB,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LinearRegression': LinearRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'SVR': SVR,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'KNeighborsRegressor': KNeighborsRegressor
        }

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Remove rows where the target column is NaN
        X = X[~y.isna()]
        y = y.dropna()

        # Impute missing values in X
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if problem_type == 'classification':
            default_models = [
                'LogisticRegression', 'RandomForestClassifier', 'SVC', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier'
            ]
            metrics = {
                'accuracy': accuracy_score,
                'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        elif problem_type == 'regression':
            default_models = [
                'LinearRegression', 'RandomForestRegressor', 'SVR', 'DecisionTreeRegressor', 'KNeighborsRegressor'
            ]
            metrics = {
                'mse': mean_squared_error,
                'mae': mean_absolute_error,
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            }
        else:
            raise ValueError("Unsupported problem type. Choose 'classification' or 'regression'.")

        if models is None:
            models = default_models
        else:
            models = set(models)

        models = set(models) - set(models_not_needed or [])
        models = [model_dict[model]() for model in models]

        all_models_with_scores = []
        best_model = None
        best_score = float('-inf') if problem_type == 'classification' else float('inf')
        best_params = {}

        for model in tqdm(models, desc="Model Evaluation"):
            model_name = type(model).__name__
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metrics[performance_metric](y_test, y_pred)

            # Store the model and its score
            all_models_with_scores.append((model_name, model, score))

            # Determine the best model
            if (problem_type == 'classification' and score > best_score) or (problem_type == 'regression' and score < best_score):
                best_model = model
                best_score = score
                best_params = model.get_params()

        return best_model, best_score, best_params, all_models_with_scores





