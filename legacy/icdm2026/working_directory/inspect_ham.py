import pandas as pd
from pathlib import Path
p=Path(r'E:\小龙虾\ham10000_novel_shortcut_screen_results\image_features.csv')
df=pd.read_csv(p)
print(df.shape)
print(df.columns.tolist())
print(df.head(2).to_string())
print('\nmetadata cols')
md=pd.read_csv(r'E:\小龙虾\HAM10000\HAM10000_metadata.csv')
print(md.shape)
print(md.columns.tolist())
print(md.head(2).to_string())
