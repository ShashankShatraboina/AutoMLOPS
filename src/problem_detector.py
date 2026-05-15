def detect_problem(df):

    # If dataset has only features (no target)
    if "target" not in df.columns:
        return "unsupervised"

    y = df["target"]

    if y.dtype == "object" or y.nunique() < 15:
        return "classification"

    return "regression"
