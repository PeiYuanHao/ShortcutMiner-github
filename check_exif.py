from pathlib import Path
from PIL import Image
root=Path(r'E:\小龙虾\HAM10000')
imgs=list((root/'HAM10000_images_part_1').glob('*.jpg'))[:3]+list((root/'HAM10000_images_part_2').glob('*.jpg'))[:3]
for p in imgs:
    im=Image.open(p)
    print(p.name, 'info_keys', list(im.info.keys()), 'exif_len', len(im.getexif()))
