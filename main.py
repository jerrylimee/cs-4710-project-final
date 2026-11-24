import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination

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

age_bins = [18, 50, 58.2, 66.2, 90]
age_labels = ["Young", "Middle", "Older", "Elderly"]

bmi_bins = [16, 22.5, 25.8, 29.5, 50]
bmi_labels = ["Low", "Normal", "High", "Very High"]

for dataset in [X_train, X_val, X_test]:
    dataset["Age"] = pd.cut(dataset["Age"], bins=age_bins,
                            labels=age_labels, include_lowest=True)
    dataset["BMI"] = pd.cut(dataset["BMI"], bins=bmi_bins,
                            labels=bmi_labels, include_lowest=True)
    dataset["Parity"] = pd.cut(dataset["Parity"], bins=[-1, 0,
                               1, 2, 3, float("inf")], labels=["0", "1", "2", "3", "4+"])

print("Discretization complete!")

print("\nTraining data:")
print(X_train.head())

for col in ["Age", "BMI", "Parity"]:
    print(f"\n{col} distribution:")
    print(X_train[col].value_counts().sort_index())

print("\nMissing values after discretization:")
print(X_train[["Age", "BMI", "Parity"]].isnull().sum())

print("\nFeature data types:")
print(X_train.dtypes)

print("\nTarget distribution:")
print(y_train.value_counts())

binary_features = [
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

for dataset in [X_train, X_val, X_test]:
    for col in binary_features:
        dataset[col] = dataset[col].astype("category")

y_train = y_train.astype("category")
y_val = y_val.astype("category")
y_test = y_test.astype("category")
print(X_train.dtypes)
print(y_train.dtypes)

# The 12 input features will be nodes in the network. (For why it's 12, refer to README.md.)
# Ovarian_Cancer_Status will be the outcome node.
# We'll initially connect each feature directly to the outcome, giving us a simple starting structure.

features = list(X_train.columns)
target = "Ovarian_Cancer_Status"
edges = [(target, feature) for feature in features]
model = DiscreteBayesianNetwork(edges)
print("Nodes:", model.nodes())
print("Edges:", model.edges())

model.fit(pd.concat([X_train, y_train], axis=1),
          estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
print("\nCPT estimation complete!")

for cpd in model.get_cpds():
    print(f"\n{cpd.variable}")
    print(cpd)

# CPT estimation complete!

# Ovarian_Cancer_Status
# +--------------------------+----------+
# | Ovarian_Cancer_Status(0) | 0.501144 |
# +--------------------------+----------+
# | Ovarian_Cancer_Status(1) | 0.498856 |
# +--------------------------+----------+

# Age
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | Age(Elderly)          | 0.18775513116953554      | 0.30940466290040436      |
# +-----------------------+--------------------------+--------------------------+
# | Age(Middle)           | 0.2506137421141275       | 0.24717530326058904      |
# +-----------------------+--------------------------+--------------------------+
# | Age(Older)            | 0.20608175615883076      | 0.2958117060021221       |
# +-----------------------+--------------------------+--------------------------+
# | Age(Young)            | 0.3555493705575062       | 0.14760832783688452      |
# +-----------------------+--------------------------+--------------------------+

# Menopausal_Status
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | Menopausal_Status(0)  | 0.3530729924923639       | 0.14535000430156864      |
# +-----------------------+--------------------------+--------------------------+
# | Menopausal_Status(1)  | 0.6469270075076361       | 0.8546499956984314       |
# +-----------------------+--------------------------+--------------------------+

# Family_History
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | Family_History(0)     | 0.9227969512717308       | 0.7199822201829601       |
# +-----------------------+--------------------------+--------------------------+
# | Family_History(1)     | 0.07720304872826925      | 0.28001777981703996      |
# +-----------------------+--------------------------+--------------------------+

# BRCA1_Status
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | BRCA1_Status(0)       | 0.9878821615140875       | 0.8236213472512977       |
# +-----------------------+--------------------------+--------------------------+
# | BRCA1_Status(1)       | 0.012117838485912477     | 0.17637865274870235      |
# +-----------------------+--------------------------+--------------------------+

# BRCA2_Status
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | BRCA2_Status(0)       | 0.9884530844109503       | 0.8587221473430644       |
# +-----------------------+--------------------------+--------------------------+
# | BRCA2_Status(1)       | 0.0115469155890497       | 0.14127785265693557      |
# +-----------------------+--------------------------+--------------------------+

# Lynch_Syndrome_Status
# +--------------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status    | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +--------------------------+--------------------------+--------------------------+
# | Lynch_Syndrome_Status(0) | 0.9954183437526762       | 0.9537007828854922       |
# +--------------------------+--------------------------+--------------------------+
# | Lynch_Syndrome_Status(1) | 0.0045816562473237985    | 0.046299217114507754     |
# +--------------------------+--------------------------+--------------------------+

# BMI
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | BMI(High)             | 0.24947189632040193      | 0.24883857646755184      |
# +-----------------------+--------------------------+--------------------------+
# | BMI(Low)              | 0.2553524021580886       | 0.25428723007656795      |
# +-----------------------+--------------------------+--------------------------+
# | BMI(Normal)           | 0.25015700379663725      | 0.24981359869232314      |
# +-----------------------+--------------------------+--------------------------+
# | BMI(Very High)        | 0.24501869772487225      | 0.2470605947635571       |
# +-----------------------+--------------------------+--------------------------+

# Parity
# +-----------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-----------------------+--------------------------+--------------------------+
# | Parity(0)             | 0.1667951243184608       | 0.16102205270855438      |
# +-----------------------+--------------------------+--------------------------+
# | Parity(1)             | 0.2965088064856841       | 0.29786928966763215      |
# +-----------------------+--------------------------+--------------------------+
# | Parity(2)             | 0.26779138477348635      | 0.27183046084138685      |
# +-----------------------+--------------------------+--------------------------+
# | Parity(3)             | 0.1611429876395193       | 0.15958819649565542      |
# +-----------------------+--------------------------+--------------------------+
# | Parity(4+)            | 0.10776169678284947      | 0.10969000028677124      |
# +-----------------------+--------------------------+--------------------------+

# Oral_Contraceptive_History
# +-------------------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status         | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +-------------------------------+--------------------------+--------------------------+
# | Oral_Contraceptive_History(0) | 0.48581256601295997      | 0.697786126007284        |
# +-------------------------------+--------------------------+--------------------------+
# | Oral_Contraceptive_History(1) | 0.51418743398704         | 0.302213873992716        |
# +-------------------------------+--------------------------+--------------------------+

# Endometriosis_History
# +--------------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status    | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +--------------------------+--------------------------+--------------------------+
# | Endometriosis_History(0) | 0.9414090377094574       | 0.8191477158670528       |
# +--------------------------+--------------------------+--------------------------+
# | Endometriosis_History(1) | 0.05859096229054266      | 0.18085228413294716      |
# +--------------------------+--------------------------+--------------------------+

# Hormone_Therapy_History
# +----------------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status      | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +----------------------------+--------------------------+--------------------------+
# | Hormone_Therapy_History(0) | 0.9023579115640433       | 0.8716842075076712       |
# +----------------------------+--------------------------+--------------------------+
# | Hormone_Therapy_History(1) | 0.09764208843595673      | 0.12831579249232886      |
# +----------------------------+--------------------------+--------------------------+

# Tubal_Ligation_History
# +---------------------------+--------------------------+--------------------------+
# | Ovarian_Cancer_Status     | Ovarian_Cancer_Status(0) | Ovarian_Cancer_Status(1) |
# +---------------------------+--------------------------+--------------------------+
# | Tubal_Ligation_History(0) | 0.7848619793896834       | 0.9215824037165553       |
# +---------------------------+--------------------------+--------------------------+
# | Tubal_Ligation_History(1) | 0.2151380206103166       | 0.07841759628344469      |
# +---------------------------+--------------------------+--------------------------+

inference = VariableElimination(model)
evidence = X_test.iloc[0].to_dict()
prediction = inference.query(variables=["Ovarian_Cancer_Status"], evidence=evidence)
print("\nExample patient:")
print(evidence)
print("\nPredicted probabilities:")
print(prediction)

y_pred_proba = []

for _, patient in X_test.iterrows():
    evidence = patient.to_dict()
    prediction = inference.query(variables=["Ovarian_Cancer_Status"], evidence=evidence, show_progress=False)
    cancer_probability = prediction.values[1]
    y_pred_proba.append(cancer_probability)

print("Predictions completed!")
print("Number of predictions:", len(y_pred_proba))
print("First 5 probabilities:", y_pred_proba[:5])