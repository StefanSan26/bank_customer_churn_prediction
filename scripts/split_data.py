import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('data/train.csv')

# Split into train and test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the files
test.to_csv('data/test.csv', index=False)
print(f'Created test set with {len(test)} samples') 