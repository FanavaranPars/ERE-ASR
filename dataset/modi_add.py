import pandas as pd
import os
from tqdm.auto import tqdm

csv_file = 'valid_split.tsv'
df = pd.DataFrame(pd.read_csv(csv_file, sep='\t', on_bad_lines='skip'))


for i, path in tqdm(enumerate(df['path'])):
    df['path'][i] = path.split('/')[-1]

df.to_csv('valid_split_1.tsv', sep='\t', index=False)