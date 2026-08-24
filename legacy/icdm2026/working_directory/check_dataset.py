import pandas as pd

metadata_path = "./synthetic_corner_patch/metadata.csv"

df = pd.read_csv(metadata_path)

print("Total samples:", len(df))
print("\nSplit counts:")
print(df["split"].value_counts())

for split in ["train", "val", "test_id", "test_ood"]:
    print("\n" + "=" * 60)
    print(split)
    print("=" * 60)

    temp = df[df["split"] == split]

    table = pd.crosstab(
        temp["class_name"],
        temp["patch_corner"],
        normalize="index"
    )

    print(table)

    print("\nAlignment ratio:")
    print(temp.groupby("class_name")["is_aligned"].mean())