import json, os, glob
import pandas as pd

stem = lambda p: os.path.splitext(os.path.basename(p))[0]

lab = pd.read_csv('shotqa_labels.csv')
lab['stem'] = lab['image'].apply(stem)

rows = []
with open('ShotQA_meta/meta.jsonl') as f:
    for line in f:
        d = json.loads(line)
        rows.append({'stem': stem(d.get('img_id') or ''), 'title': d.get('title'),
                     'frame_size': d.get('Frame Size'), 'shot_type': d.get('Shot Type')})
meta = pd.DataFrame(rows)
print('meta:', len(meta), 'imagens |', meta['title'].nunique(), 'títulos')

ss = lab[lab['question'].str.contains('shot size')].merge(meta, on='stem', how='left')
print('\nshot size com meta:', ss['title'].notna().sum(), 'de', len(ss))
com = ss[ss['frame_size'].notna()]
ok = com['label'].str.strip() == com['frame_size'].str.strip()
print('etiqueta == Frame Size:', ok.sum(), 'de', len(com))
print(com.loc[~ok, ['label', 'frame_size']].head(8))
print('\nfilmes distintos (shot size):', ss['title'].nunique())
print(ss['title'].value_counts().head(5))

sb = {stem(p) for p in glob.glob('ShotBench/image/*')}
m = meta[meta['stem'].isin(sb)]
films_test = set(m['title'].dropna())
print('\nimagens do ShotBench encontradas no meta:', len(m), 'de', len(sb))
print('filmes do ShotBench identificados:', len(films_test))
print('imagens de shot size no ShotQA de filmes que estão no ShotBench:', ss['title'].isin(films_test).sum())

g = json.load(open('ShotQA_meta/grpo.json'))
gs = {x['images'][0] for x in g if x.get('_category') == 'shot size'}
print('\ngrpo shot size:', len(gs), '| já no sft:', len(gs & set(lab['image'])))

for name, key in [('shot_size', 'shot size'), ('framing', 'shot framing'), ('camera_angle', 'camera angle')]:
    imgs = sorted(lab[lab['question'].str.contains(key)]['image'].unique())
    with open(f'lista_{name}_tar.txt', 'w') as f:
        f.write('\n'.join('./' + os.path.basename(p) for p in imgs) + '\n')
    print(name, len(imgs), 'imagens')
