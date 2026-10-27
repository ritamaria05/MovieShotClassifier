# build_dataset.py
import os, json, ast, re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
norm = lambda s: re.sub(r'\s+', ' ', str(s).lower().replace('-', ' ')).strip()

DIMS = {'shot_size': 'shot size', 'framing': 'shot framing'}   # nome exato da category

# ---------- meta: stem -> filme ----------
rows = [json.loads(l) for l in open('ShotQA_meta/meta.jsonl')]
meta = pd.DataFrame([(stem(d.get('img_id') or ''), d.get('title') or '') for d in rows],
                    columns=['stem', 'title']).drop_duplicates('stem')
meta['film'] = meta['title'].str.split('/').str[0].str.strip()

# ---------- teste (ShotBench) ----------
ann = pd.read_csv('ShotBench/annotations.csv')
print('categorias no ShotBench:', ann['category'].unique().tolist())
ann['img'] = ann['path'].apply(lambda s: ast.literal_eval(s)[0])
ann['label'] = [json.loads(o)[a] for o, a in zip(ann['options'], ann['answer'])]
ann['stem'] = ann['img'].apply(stem)
ann['path'] = ann['img'].apply(lambda p: os.path.join('ShotBench', p))
ann = ann.merge(meta[['stem', 'film']], on='stem', how='left')

films_test = set(ann['film'].dropna())
print('imagens de teste sem filme identificado:', ann['film'].isna().sum(), 'de', len(ann))

os.makedirs('splits', exist_ok=True)
for name, kw in DIMS.items():
    # --- teste
    t = ann[ann['category'].str.lower() == kw].copy()          # igualdade, não contains
    t['label'] = t['label'].apply(norm)
    t['multi'] = t['label'].str.contains(',')
    t['exists'] = t['path'].apply(os.path.exists)
    print(f'\n=== {name} ===\nteste: {len(t)} | multi-etiqueta: {t["multi"].sum()} | em falta: {(~t["exists"]).sum()}')
    t = t[t['exists']][['path', 'label', 'multi']]
    t.to_csv(f'splits/test_{name}.csv', index=False)

    # --- treino
    df = pd.read_csv(f'labels_{name}.csv')
    df = df[~df['multi']].copy()
    df['label'] = df['label'].apply(norm)
    df['path'] = df['image'].apply(lambda p: os.path.join('data_shotqa', os.path.basename(p)))
    n0 = len(df)
    leak = df['film'].isin(films_test)
    df = df[~leak]
    print(f'treino: {n0} (só etiqueta única) -> {len(df)}')

    # vocabulário treino vs teste
    ctr, cte = set(df['label']), set(t['label'])
    print('classes só no treino:', ctr - cte, '| só no teste:', cte - ctr)

    # split treino/val por filme (sem filme -> grupo próprio)
    groups = df['film'].fillna(df['stem'])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr, va = next(gss.split(df, groups=groups))
    df.iloc[tr][['path', 'label', 'film']].to_csv(f'splits/train_{name}.csv', index=False)
    df.iloc[va][['path', 'label', 'film']].to_csv(f'splits/val_{name}.csv', index=False)

    print('treino/val:', len(tr), '/', len(va))
    print(pd.DataFrame({'train': df.iloc[tr]['label'].value_counts(),
                        'val': df.iloc[va]['label'].value_counts(),
                        'test': t['label'].value_counts()}).fillna(0).astype(int))