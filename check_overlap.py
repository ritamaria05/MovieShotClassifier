import glob, os, json
import pandas as pd

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
sb = {stem(p) for p in glob.glob('ShotBench/image/*')}
print('imagens do ShotBench no disco:', len(sb))

lab = pd.read_csv('labels_shot_size.csv')
print('em comum com o treino (pelo nome):', len(sb & set(lab['stem'])))

rows = [json.loads(l) for l in open('ShotQA_meta/meta.jsonl')]
meta = pd.DataFrame([{'stem': stem(d.get('img_id') or ''), 'title': d.get('title') or ''} for d in rows])
meta['film'] = meta['title'].str.split('/').str[0].str.strip()
m = meta[meta['stem'].isin(sb)]
films = set(m['film'])
print('ShotBench encontradas no meta:', len(m), '| filmes:', len(films))
print('imagens de treino nesses filmes:', lab['film'].isin(films).sum(), 'de', len(lab))
