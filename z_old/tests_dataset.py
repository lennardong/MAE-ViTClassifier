"""
Question:
is concatenating multiple datasets and subsequently splitting them creating a downstream issue?

Test:
1. create a standard dataset (dsbase)
2. create a standard dataset, then split it (ds3)
3. create a concatenated dataset, then split it (ds)

Conclusion:
all 3 return the same object class instance.

>>> dsbase
DatasetDict({
    train: Dataset({
        features: ['image'],
        num_rows: 108
    })
})

>>> ds3
DatasetDict({
    train: Dataset({
        features: ['image'],
        num_rows: 81
    })
    validation: Dataset({
        features: ['image'],
        num_rows: 27
    })
})

>>> ds
DatasetDict({
    train: Dataset({
        features: ['image'],
        num_rows: 162
    })
    validation: Dataset({
        features: ['image'],
        num_rows: 54
    })
})
"""
from datasets import load_dataset, concatenate_datasets, DatasetDict

#Baseline
dsbase = load_dataset("imagefolder", data_dir='./data/CAM16_100cls_10mask/test/data/normal')

# Baseline + Scripted Split
ds3 = load_dataset("imagefolder", data_dir='./data/CAM16_100cls_10mask/test/data/normal')
split3 = ds3["train"].train_test_split(0.25)
ds3["train"] = split3["train"]
ds3["validation"] = split3["test"]

# Concatenation + Scripted Split
ds1 = load_dataset("imagefolder", data_dir='./data/CAM16_100cls_10mask/test/data/normal', split = "train")
ds2 = load_dataset("imagefolder", data_dir='./data/CAM16_100cls_10mask/test/data/normal', split = "train")
ds = concatenate_datasets([ds1, ds2])
ds = DatasetDict({"train": ds})

split = ds["train"].train_test_split(0.25)
ds["train"] = split["train"]
ds["validation"] = split["test"]
