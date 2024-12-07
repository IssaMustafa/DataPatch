import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from src.exceptions import NotEnoughRows
from src.train_and_evaluate import TrainAndEvaluate


class FillNull:
    """
    Class for handling missing values in a DataFrame and training models to impute them.

    Attributes:
        df (pd.DataFrame): The input DataFrame.
        null_columns (list): List of columns with missing values.
        label_encoders (dict): LabelEncoders for categorical columns.
        imputation_info (dict): Information about the imputation process.
    """

    def __init__(self, df):
        """
        Initializes the FillNull object.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df
        self.null_columns = self.is_null()
        self.label_encoders = {}
        self.imputation_info = {}
        self.encode_labels()

    def determine_target_type(self, target_column):
        """
        Determines if a target column is for classification or regression.

        Args:
            target_column (str): The target column name.

        Returns:
            str: 'classification' or 'regression'.
        """
        target_dtype = self.df[target_column].dtype
        if pd.api.types.is_numeric_dtype(target_dtype):
            unique_values = self.df[target_column].nunique()
            value_counts = self.df[target_column].value_counts(normalize=True)
            if unique_values < 10:
                return 'classification'
            elif unique_values >= 10 and unique_values <= 20:
                if any(value_counts > 0.1):
                    return 'classification'
                else:
                    return 'regression'
            else:
                if value_counts.iloc[0] > 0.05:
                    return 'classification'
                else:
                    return 'regression'
        else:
            return 'classification'

    def is_null(self):
        """
        Identifies columns with missing values.

        Returns:
            list: List of column names with missing values.
        """
        return [i for i in self.df.isnull().sum().index if self.df.isnull().sum()[i] > 0]

    def encode_labels(self):
        """
        Encodes categorical columns in the DataFrame.
        """
        object_columns = self.df.select_dtypes(include=['object']).columns
        for col in object_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

    def check_missing_data(self):
        """
        Checks if missing data exceeds a threshold and raises an exception if too few rows remain.
        """
        for col in self.null_columns:
            missing_percentage = self.df[col].isnull().sum() / self.df.shape[0]
            if missing_percentage > 0.6:
                remaining_rows = (1 - missing_percentage) * self.df.shape[0]
                if remaining_rows < 1000:
                    raise NotEnoughRows(f"Not enough rows in the data after dropping the nulls for column {col}.")


    def fill_missing_values_and_train(self, test_size=0.25, models=None, hyperparameter_tuning=False, 
                                    hyperparameter_tuning_threshold=0.5, n_jobs=-1, models_not_needed=None, 
                                    performance_metric='accuracy'):
        """
        Handles missing values by training models to predict them and storing all trained models.

        Returns:
            pd.DataFrame: DataFrame with missing values filled (encoded).
        """
        self.check_missing_data()
        self.trained_models = {}  # Dictionary to store all trained models for each column

        for col in self.null_columns:
            print(f"Processing column: {col}")
            problem_type = self.determine_target_type(col)
            performance_metric = 'accuracy' if problem_type == 'classification' else 'rmse'

            trainer = TrainAndEvaluate(self.df)
            try:
                best_model, best_score, best_params, all_models_with_scores = trainer.train_and_evaluate(
                    target_column=col,
                    test_size=test_size,
                    models=models,
                    problem_type=problem_type,
                    hyperparameter_tuning=hyperparameter_tuning,
                    hyperparameter_tuning_threshold=hyperparameter_tuning_threshold,
                    n_jobs=n_jobs,
                    models_not_needed=models_not_needed,
                    performance_metric=performance_metric,
                )

                # Store all trained models for the column
                self.trained_models[col] = all_models_with_scores

                # Predict missing values
                missing_data = self.df[self.df[col].isnull()]
                if not missing_data.empty:
                    imputer = SimpleImputer(strategy='mean')
                    missing_data_imputed = pd.DataFrame(imputer.fit_transform(missing_data.drop(columns=[col])), columns=missing_data.drop(columns=[col]).columns)
                    predictions = best_model.predict(missing_data_imputed)
                    self.df.loc[self.df[col].isnull(), col] = predictions

                self.imputation_info[col] = {
                    'model': type(best_model).__name__,
                    'best_metric': performance_metric,
                    'best_score': best_score,
                }

            except Exception as e:
                print(f"Error in processing {col}: {e}")
                fallback_imputer = SimpleImputer(strategy='most_frequent' if problem_type == 'classification' else 'mean')
                self.df[col] = fallback_imputer.fit_transform(self.df[[col]])

        # Print the imputation info
        for col, info in self.imputation_info.items():
            print(f"{col}: {{'model': '{info['model']}', 'best_metric': '{info['best_metric']}', 'best_score': {info['best_score']}}}")

        return self.df





