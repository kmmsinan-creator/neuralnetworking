"""process_data.py
python process_data.py
Requires: pandas
"""
import pandas as pd
import json


# Paths (assumes these CSVs are in the same folder)
TRAIN = "train.csv"
TEST = "test.csv"
OUT = "processed_data.json"


def load_data():
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
return train, test


def make_aggregates(train):
# Basic aggregates and grouped stats used by the webpage
aggregates = {}


aggregates['survival_counts'] = train['Survived'].value_counts().to_dict()


# Survival rate by sex
aggregates['survival_by_sex'] = train.groupby('Sex')['Survived'].mean().round(4).to_dict()


# Survival rate by Pclass
aggregates['survival_by_pclass'] = train.groupby('Pclass')['Survived'].mean().round(4).to_dict()


# Survival rate by Embarked (drop NaNs)
aggregates['survival_by_embarked'] = train.dropna(subset=['Embarked']).groupby('Embarked')['Survived'].mean().round(4).to_dict()


# Age distributions split by survival for histogram plotting
ages_survived = train[train['Survived']==1]['Age'].dropna().tolist()
ages_not = train[train['Survived']==0]['Age'].dropna().tolist()
aggregates['age_by_survival'] = {
'survived': ages_survived,
'not_survived': ages_not
}


# Fare distribution (simple list)
aggregates['fare'] = train['Fare'].dropna().tolist()


# Save a small sample table (first 50 rows) for quick display if wanted
aggregates['sample_rows'] = train.head(50).to_dict(orient='records')


return aggregates


if __name__ == '__main__':
print('Loading data...')
train, test = load_data()
print('Computing aggregates...')
data = make_aggregates(train)
print(f'Writing {OUT}...')
with open(OUT, 'w') as f:
json.dump(data, f)
print('Done. Created processed_data.json')
