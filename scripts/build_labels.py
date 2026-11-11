import json, re, os, glob
import pandas as pd

data = json.load(open('ShotQA_meta/sft.json'))
q_re = re.compile(r"Question:\s*(.+?)\nOptions:\n(.+?)\nPlease select", re.S)
opt_re = re.compile(r"^([A-H])\.\s*(.+)$", re.M)

rows, skipped = [], 0
for item in data:
    imgs = item.get('images') or []
    if len(imgs) != 1:
        continue
    msgs = item['messages']
    for i in range(0, len(msgs) - 1, 2):
        u, a = msgs[i], msgs[i + 1]
        if u['role'] != 'user' or a['role'] != 'assistant':
            continue
        m = q_re.search(u['content'])
        if not m:
            skipped += 1
            continue
        opts = dict(opt_re.findall(m.group(2)))
        letter = a['content'].strip()
        if letter in opts:
            rows.append((imgs[0], m.group(1).strip(), opts[letter].strip()))
        else:
            skipped += 1

df = pd.DataFrame(rows, columns=['image', 'question', 'label'])
df['dimension'] = df['question'].str.extract(r"What'?s the (.+?) of this shot", expand=False)
df.to_csv('shotqa_labels.csv', index=False)

print(len(data), 'conversas |', df['image'].nunique(), 'imagens |', len(df), 'perguntas | ignoradas:', skipped)
print('\nperguntas mais comuns:\n', df['question'].value_counts().head(12))

ss = df[df['question'].str.contains('shot size', case=False)]
print('\nshot size:', len(ss), 'perguntas |', ss['image'].nunique(), 'imagens')
print(ss['label'].value_counts())

piv = df.assign(v=1).pivot_table(index='image', columns='dimension', values='v', aggfunc='max', fill_value=0)
print('\nimagens por dimensão:\n', piv.sum())
print('\nimagens com ambas as dimensões:\n', piv.T.dot(piv))

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
sb = {stem(p) for p in glob.glob('ShotBench/image/*')}
qa = {stem(p) for p in df['image']}
print('\nimagens em comum com o ShotBench:', len(sb & qa))

open('lista_shot_size.txt', 'w').write('\n'.join(sorted(ss['image'].unique())) + '\n')

g = json.load(open('ShotQA_meta/grpo.json'))
print('\ngrpo.json:', type(g), len(g))
print(json.dumps(g[0], ensure_ascii=False)[:600])
