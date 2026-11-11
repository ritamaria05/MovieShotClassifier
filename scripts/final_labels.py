import json, os
import pandas as pd

DIMS = {'shot_size': 'shot size', 'framing': 'shot framing', 'camera_angle': 'camera angle'}
USE = ['shot_size','framing']   # acrescenta 'framing' e/ou 'camera_angle' se quiseres

stem = lambda p: os.path.splitext(os.path.basename(p))[0]

rows = []
with open('ShotQA_meta/meta.jsonl') as f:
    for line in f:
        d = json.loads(line)
        rows.append((stem(d.get('img_id') or ''), d.get('title') or ''))
meta = pd.DataFrame(rows, columns=['stem', 'title']).drop_duplicates('stem')
meta['film'] = meta['title'].str.split('/').str[0].str.strip()

lab = pd.read_csv('shotqa_labels.csv')
g = json.load(open('ShotQA_meta/grpo.json'))

all_list = set()
for name in USE:
    key = DIMS[name]
    a = lab[lab['question'].str.contains(key)][['image', 'label']].copy()
    a['source'] = 'sft'
    b = pd.DataFrame([(x['images'][0], ', '.join(x['_labels']), 'grpo')
                      for x in g if key in str(x.get('_category', '')).lower()],
                     columns=['image', 'label', 'source'])
    df = pd.concat([a, b]).drop_duplicates('image')
    df['stem'] = df['image'].apply(stem)
    df = df.merge(meta, on='stem', how='left')
    df['multi'] = df['label'].str.contains(',')
    df.to_csv(f'labels_{name}.csv', index=False)
    all_list |= {'./' + os.path.basename(p) for p in df['image']}
    print(name, len(df), 'imagens |', (~df['multi']).sum(), 'com uma só etiqueta |',
          df['film'].nunique(), 'filmes | sem meta:', df['film'].isna().sum())

open('lista_extrair.txt', 'w').write('\n'.join(sorted(all_list)) + '\n')
print('total a extrair:', len(all_list))
