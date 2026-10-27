# verify.py
import os, pandas as pd

for name in ['shot_size', 'framing']:
    df = pd.read_csv(f'labels_{name}.csv')
    df['path'] = df['image'].apply(lambda p: os.path.join('data_shotqa', os.path.basename(p)))
    exists = df['path'].apply(os.path.exists)
    print(name, '| no CSV:', len(df), '| no disco:', exists.sum(), '| em falta:', (~exists).sum())

print('ficheiros em data_shotqa:', len(os.listdir('data_shotqa')))
print('linhas em lista_extrair.txt:', sum(1 for _ in open('lista_extrair.txt')))

ann = pd.read_csv('ShotBench/annotations.csv')
print(ann.shape); print(ann.columns.tolist()); print(ann.head(3).T)