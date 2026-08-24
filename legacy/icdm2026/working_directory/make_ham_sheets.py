from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
root=Path(r'E:\小龙虾\HAM10000')
res=Path(r'E:\小龙虾\ham10000_novel_shortcut_screen_results')
out=Path(r'E:\小龙虾\ham10000_qualitative_samples')
out.mkdir(exist_ok=True)
df=pd.read_csv(res/'image_features.csv')
# locate images
paths={}
for part in ['HAM10000_images_part_1','HAM10000_images_part_2']:
    for p in (root/part).glob('*.jpg'):
        paths[p.stem]=p

groups={
 'high_saturation': df.sort_values('mean_S', ascending=False).head(30),
 'high_brightness': df.sort_values('mean_V', ascending=False).head(30),
 'low_brightness': df.sort_values('mean_V', ascending=True).head(30),
 'high_dark_corner': df.sort_values('dark_corner_frac', ascending=False).head(30),
 'high_dark_border': df.sort_values('dark_border_frac', ascending=False).head(30),
 'high_border_center_l2': df.sort_values('border_center_l2', ascending=False).head(30),
 'high_file_size': df.sort_values('file_size', ascending=False).head(20),
 'low_file_size': df.sort_values('file_size', ascending=True).head(20),
}

def thumb(path, size=(180,135)):
    im=Image.open(path).convert('RGB')
    im.thumbnail(size)
    canvas=Image.new('RGB', size, 'white')
    canvas.paste(im, ((size[0]-im.width)//2,(size[1]-im.height)//2))
    return canvas

for name, sub in groups.items():
    thumbs=[]; labels=[]
    for _,r in sub.iterrows():
        p=paths.get(r.image_id)
        if not p: continue
        thumbs.append(thumb(p))
        labels.append(f"{r.dx} {r.image_id}\nS{r.mean_S:.0f} V{r.mean_V:.0f}\ndb{r.dark_border_frac:.2f} dc{r.dark_corner_frac:.2f}")
    cols=5; tw,th=(180,135); lh=44
    rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new('RGB',(cols*tw, rows*(th+lh)), (245,245,245))
    draw=ImageDraw.Draw(sheet)
    for i,(im,lab) in enumerate(zip(thumbs,labels)):
        r,c=divmod(i,cols); x=c*tw; y=r*(th+lh)
        sheet.paste(im,(x,y)); draw.text((x+4,y+th+2),lab,fill=(0,0,0))
    path=out/f'{name}_sheet.jpg'
    sheet.save(path, quality=92)
    print(path)
# write sample lists
with open(out/'sample_lists.txt','w',encoding='utf-8') as f:
    for name,sub in groups.items():
        f.write(f'\n== {name} ==\n')
        f.write(sub[['image_id','dx','part','file_index','file_size','mean_S','mean_V','dark_border_frac','dark_corner_frac','border_center_l2']].to_string(index=False))
        f.write('\n')
