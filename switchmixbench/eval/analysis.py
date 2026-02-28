def robustness_drop(df_clean, df_mix, label_col="label", pred_col="prediction"):
    def acc(df):
        return (df[pred_col].astype(str).str.lower().str.strip() == df[label_col].astype(str).str.lower().str.strip()).mean()
    return acc(df_clean) - acc(df_mix)
