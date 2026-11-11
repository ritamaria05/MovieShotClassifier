from datasets import load_dataset
from paths import CSV_DIR, ensure_output_dirs

ensure_output_dirs()
df = load_dataset('Vchitect/ShotBench', split='test').to_pandas()
df.to_csv(CSV_DIR / 'annotations.csv', index=False)
print(df.shape)
