#create gold version tsv file
import pandas as pd

train = 'data/ntu-train-track2-uncovered.txt'
gold = 'data/ntu.train.gold.tsv'

col1 = []
col2 = []

with open(train, 'r') as r:
  for line in r:
    if line[0] == '\\' and line[1] == 't':
      col1.append(line[3:].rstrip())
    elif line[0] == '\\' and line[1] == 'm':
      col2.append(line[3:].replace('-', ' @@').rstrip())

df = pd.DataFrame({'col1': col1, 'col2': col2})
df.to_csv(gold, sep='\t', header=None, index=False)
