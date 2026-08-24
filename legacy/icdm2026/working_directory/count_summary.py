from pathlib import Path
from collections import defaultdict
root = Path(r'E:\小龙虾\waterbirds')
# counts
counts=[]
for p in sorted([d for d in root.iterdir() if d.is_dir()]):
    n=sum(1 for _ in p.glob('*.jpg'))+sum(1 for _ in p.glob('*.jpeg'))+sum(1 for _ in p.glob('*.png'))
    counts.append((p.name,n))
print('min_count', min(n for _,n in counts))
print('max_count', max(n for _,n in counts))
print('avg_count', sum(n for _,n in counts)/len(counts))
for name,n in counts[:10]:
    pass
# show selected group counts
for prefix in ['059.California_Gull','060.Glaucous_winged_Gull','061.Heermann_Gull','062.Herring_Gull','064.Ring_billed_Gull','066.Western_Gull',
               '141.Artic_Tern','142.Black_Tern','143.Caspian_Tern','144.Common_Tern','145.Elegant_Tern','146.Forsters_Tern','147.Least_Tern',
               '158.Bay_breasted_Warbler','160.Black_throated_Blue_Warbler','169.Magnolia_Warbler','171.Myrtle_Warbler','182.Yellow_Warbler']:
    p=root/prefix
    n=len(list(p.glob('*.jpg')))+len(list(p.glob('*.jpeg')))+len(list(p.glob('*.png')))
    print(prefix,n)
