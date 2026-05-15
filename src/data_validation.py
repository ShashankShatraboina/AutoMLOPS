def validate_data(df):

    if df.empty:
        raise ValueError("Dataset is empty")

    print("✅ Data validation passed")
