import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def preprocessing():
    df_raw = pd.read_csv('../data/raw/train.csv')

    df_raw.drop("id", axis=1, inplace=True)
    df_raw.drop(df_raw[df_raw["person_emp_length"] == 123].index, inplace=True)
    df_raw.drop(df_raw[df_raw["person_age"] == 123].index, inplace=True)
    df_raw.drop(df_raw[df_raw["person_emp_length"] > df_raw["person_age"] - 14].index, inplace=True)

    df_raw = encode_with_one_hot_encoder(df_raw, "person_home_ownership")
    df_raw = encode_with_one_hot_encoder(df_raw, "loan_intent")

    df_raw = encode_with_ordinal_encoder(df_raw, "loan_grade", [['A', 'B', 'C', 'D', 'E', 'F', 'G']])
    df_raw = encode_with_ordinal_encoder(df_raw, "cb_person_default_on_file", [['N', 'Y']])

    df_raw['person_income'] = np.log1p(df_raw['person_income'])
    df_raw['loan_amnt'] = np.log1p(df_raw['loan_amnt'])

    #df_raw.drop("person_age", axis=1, inplace=True)
    #df_raw.drop("loan_int_rate", axis=1, inplace=True)

    corr = df_raw.corr(numeric_only=True)
    sns.heatmap(corr,
                annot=True,  # Show correlation values
                fmt='.2f',  # Format to 2 decimal places
                cmap='RdBu_r',  # Better color scheme
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.savefig("../images/correlation_heatmap.png", dpi=300, bbox_inches="tight")

    df_raw.to_csv('../data/processed/train.csv', index=False)

def encode_with_one_hot_encoder(df, column_name):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_column = one_hot_encoder.fit_transform(df[[column_name]])
    new_columns = one_hot_encoder.get_feature_names_out([column_name])
    df_encoded_home_ownership = pd.DataFrame(encoded_column, columns=new_columns, index=df.index)
    return df.drop(columns=column_name).join(df_encoded_home_ownership)

def encode_with_ordinal_encoder(df, column_name, categories):
    ordinal_encoder = OrdinalEncoder(categories=categories)
    df[column_name] = ordinal_encoder.fit_transform(df[[column_name]])
    return df


if __name__ == "__main__":
    preprocessing()
