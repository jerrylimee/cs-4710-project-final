import pandas as pd

df = pd.read_csv("data/processed/oc_risk50k_selected.csv")

train_df = df[df["Partition"] == "Train"].copy()
val_df = df[df["Partition"] == "Validation"].copy()
test_df = df[df["Partition"] == "Test"].copy()

X_train = train_df.drop(columns=["Ovarian_Cancer_Status", "Partition"])
y_train = train_df["Ovarian_Cancer_Status"]

X_val = val_df.drop(columns=["Ovarian_Cancer_Status", "Partition"])
y_val = val_df["Ovarian_Cancer_Status"]

X_test = test_df.drop(columns=["Ovarian_Cancer_Status", "Partition"])
y_test = test_df["Ovarian_Cancer_Status"]

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nAge distribution:")
print(X_train["Age"].describe())

# Age has a fairly broad distribution. I suggest dividing it into four categories using quartiles.

print("\nBMI distribution:")
print(X_train["BMI"].describe())

# We can initially use the same quartile-based approach.

print("\nParity distribution:")
print(X_train["Parity"].value_counts())

# I suggest grouping parity into five categories because having separate categories for 7 and 8 would create very sparsely populated states in our Bayesian Network.

for col in ["Age", "BMI"]:
    _, boundaries = pd.qcut(X_train[col], q=4, retbins=True, duplicates="drop")
    print(f"\n{col} boundaries:")
    print(boundaries)
