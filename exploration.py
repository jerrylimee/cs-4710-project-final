import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("data/oc_risk50k_single_dataset.csv")

print("Dataset dimensions:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nColumn information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nCancer status distribution:")
print(df["Ovarian_Cancer_Status"].value_counts())

print("\nCancer status percentages:")
print(df["Ovarian_Cancer_Status"].value_counts(normalize=True) * 100)

print("\nCancer distribution by partition:")
print(pd.crosstab(df["Partition"], df["Ovarian_Cancer_Status"], margins=True))

print("\nCancer percentages by partition:")
print(pd.crosstab(df["Partition"],
      df["Ovarian_Cancer_Status"], normalize="index") * 100)

features = [
    "Age",
    "Menopausal_Status",
    "Family_History",
    "BRCA1_Status",
    "BRCA2_Status",
    "Lynch_Syndrome_Status",
    "BMI",
    "Parity",
    "Oral_Contraceptive_History",
    "Endometriosis_History",
    "Hormone_Therapy_History",
    "Tubal_Ligation_History"
]

print("\nFeature summary:")
print(df[features].describe().T)

categorical_features = [
    "Menopausal_Status",
    "Family_History",
    "BRCA1_Status",
    "BRCA2_Status",
    "Lynch_Syndrome_Status",
    "Oral_Contraceptive_History",
    "Endometriosis_History",
    "Hormone_Therapy_History",
    "Tubal_Ligation_History"
]

for col in categorical_features:
    print(f"\n{col}:")
    print(df[col].value_counts().sort_index())

for col in categorical_features:
    print(f"\n{col}:")
    result = pd.crosstab(
        df[col], df["Ovarian_Cancer_Status"], normalize="index") * 100
    print(result.round(2))

# A few observations:
# - BRCA1, BRCA2, and Lynch syndrome: Extremely strong associations with the target.
# - Family history: Also a substantial association.
# - Endometriosis: A notable association.
# - Oral contraceptive history and tubal ligation: Their category-1 groups have lower cancer rates.
# - Hormone therapy: A comparatively modest association.

numerical_features = ["Age", "BMI", "Parity"]
print("\nNumerical features by cancer status:")
print(df.groupby("Ovarian_Cancer_Status")[
      numerical_features].describe().round(2))

# The cancer group is approximately 6.3 years older on average compared to the group without cancer.
# Parity might contribute relatively little predictive information.

print("\nAge, BMI, and Parity by cancer status:")
print(df.groupby("Ovarian_Cancer_Status")[numerical_features].agg(
    ["mean", "median", "std"]).round(2).T)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col in zip(axes, numerical_features):
    sns.boxplot(data=df, x="Ovarian_Cancer_Status", y=col, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Cancer Status (0 = No, 1 = Yes)")

plt.tight_layout()
plt.show()

columns = features + ["Ovarian_Cancer_Status", "Partition"]

df_processed = df[columns].copy()

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df_processed.to_csv(output_dir / "oc_risk50k_selected.csv", index=False)

print("\nProcessed dataset:")
print(df_processed.shape)

print("\nSaved successfully!")