import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

MANIFEST = Path(r"C:\SlickTrace\outputs\dataset_manifest.csv")
OUT_DIR = Path(r"C:\SlickTrace\outputs")
SEED = 42

df = pd.read_csv(MANIFEST)

train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED,
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=SEED,
)

train = train.copy()
val = val.copy()
test = test.copy()

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"

full = pd.concat([train, val, test], ignore_index=True)

full.to_csv(OUT_DIR / "dataset_splits.csv", index=False)
train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR / "val.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)

print("SPLIT COMPLETE")
print("Seed:", SEED)
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print()
print("Train distribution:")
print(train["label"].value_counts())
print()
print("Validation distribution:")
print(val["label"].value_counts())
print()
print("Test distribution:")
print(test["label"].value_counts())