import pandas as pd

# load data
fake = pd.read_csv("../news/Fake.csv")
true = pd.read_csv("../news/True.csv")

# drop non necessary columns
cols = ["title", "subject", "date"]

true = true.drop(cols, axis=1)
fake = fake.drop(cols, axis=1)

# add label column
true_col = []
fake_col = []

for i in range(len(true)):
    true_col.append(1)

for i in range(len(fake)):
    fake_col.append(0)

true["label"] = true_col
fake["label"] = fake_col

# concat and shuffle into one dataset
dataset = pd.concat([true, fake], ignore_index=True)

dataset = dataset.sample(frac=1).reset_index(drop=True)


# Create columns needed for the BERT algorithm
# Arange columns in the correct order for BERT

throwaway = []

for i in range(len(dataset)):
    throwaway.append("A")

dataset["throwaway"] = throwaway
dataset["id"] = dataset.index + 1 

dataset = dataset[["id", "label", "throwaway", "text"]]

# Prepare tsv files and splits
# train & dev tsv should look like ( id, label, throw, text)
# test tsv = (id, text)

split_one = dataset.sample(frac=0.9, random_state=41)
test = dataset.drop(split_one.index)

train = split_one.sample(frac=0.9, random_state=41)
dev = split_one.drop(train.index)

# Train = 36367 ish samples
# Dev = 4041 ish samples
# test = 4490 ish samples

# drop label and throw columns from test
test = test.drop(["label", "throwaway"], axis=1)

# create one tsv file for each split

train.to_csv('train.tsv', sep='\t', index=False, header=False)
dev.to_csv('dev.tsv', sep='\t', index=False, header=False)
test.to_csv('test.tsv', sep='\t', index=False, header=False)

