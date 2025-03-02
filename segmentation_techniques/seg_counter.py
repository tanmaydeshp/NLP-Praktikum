import pandas as pd

f = 'outputs/git.sentence.test.bpe_guess.tsv'

df = pd.read_csv(f,sep='\t',header=None)
series = df[1].astype(str)
i = 0
for s in series:
  i = i + s.count('@@')

print(i)