import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def extract_data(file_path):
    print("Extracting data...")
    return pd.read_csv(file_path)

def transform_data(df):
    print("Transforming data...")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    transformed_data = preprocessor.fit_transform(df)

    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_features)

    return pd.DataFrame(transformed_data, columns=feature_names)

def load_data(df, output_path):
    print(f"Loading data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("ETL process completed successfully!")


if __name__ == "__main__":
    input_file = 'raw_data.csv'   
    output_file = 'processed_data.csv'

    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found. Please check the file path.")
    else:
        raw_data = extract_data(input_file)
        processed_data = transform_data(raw_data)
        load_data(processed_data, output_file)
