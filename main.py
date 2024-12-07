import pandas as pd
from src.fill_null import FillNull

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("your_dataset.csv")

    # Instantiate the FillNull class and process the dataset
    fill_null = FillNull(df)
    
    # Fill missing values and train models
    filled_df = fill_null.fill_missing_values_and_train()
    print("The null values are:")
    print(filled_df.isnull().sum())

    # Access all trained models for a specific column, e.g., 'Engine'
    trained_models_for_your_coulmn = fill_null.trained_models.get('Any column name')
    print("Trained models for 'your column':")
    print(trained_models_for_your_coulmn)

    # Save the processed DataFrame
    filled_df.to_csv("processed_dataset.csv", index=False)
