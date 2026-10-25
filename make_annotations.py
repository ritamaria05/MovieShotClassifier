from datasets import load_dataset
df = load_dataset('Vchitect/ShotBench', split='test').to_pandas()
df.to_csv('ShotBench/annotations.csv', index=False)
print(df.shape)
