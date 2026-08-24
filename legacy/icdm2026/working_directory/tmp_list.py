from pathlib import Path
root = Path(r'E:\小龙虾\waterbirds')
classes = [p for p in root.iterdir() if p.is_dir()]
print('num_classes', len(classes))
for p in classes[:8]:
    imgs = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png'))
    print(p.name, len(imgs), [x.name for x in imgs[:3]])
