# verify.py
import os, pandas as pd
from paths import CSV_DIR, SHOTBENCH_DIR, SHOTQA_IMAGES_DIR, TXT_DIR, ensure_output_dirs

ensure_output_dirs()

for name in ['shot_size', 'framing']:
    df = pd.read_csv(CSV_DIR / f'labels_{name}.csv')
    df['path'] = df['image'].apply(
        lambda p: os.path.join(SHOTQA_IMAGES_DIR, os.path.basename(p))
    )
    exists = df['path'].apply(os.path.exists)
    print(name, '| no CSV:', len(df), '| no disco:', exists.sum(), '| em falta:', (~exists).sum())

print('ficheiros em data_shotqa:', len(os.listdir(SHOTQA_IMAGES_DIR)))
print('linhas em lista_extrair.txt:', sum(1 for _ in open(TXT_DIR / 'lista_extrair.txt')))

ann = pd.read_csv(CSV_DIR / 'annotations.csv')
print(ann.shape); print(ann.columns.tolist()); print(ann.head(3).T)