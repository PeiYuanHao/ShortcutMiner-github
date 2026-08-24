import csv
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(r'E:\小龙虾\HAM10000')
features = Path(r'E:\小龙虾\ham10000_novel_shortcut_screen_results\image_features.csv')
out_dir = Path(r'C:\Users\PeiYuan Hao\.openclaw\workspace-summarize_agent\reports')
out_dir.mkdir(parents=True, exist_ok=True)

rows=[]
with features.open('r', encoding='utf-8', newline='') as f:
    reader=csv.DictReader(f)
    for r in reader:
        try:
            r['_dark_border']=float(r.get('dark_border_frac') or 0)
            r['_dark_corner']=float(r.get('dark_corner_frac') or 0)
            r['_mean_v']=float(r.get('mean_V') or 0)
        except Exception:
            continue
        rows.append(r)

# prefer mel/bkl high dark border/corner since key risk group
rows_sorted=sorted(rows, key=lambda r: (r['_dark_border']+r['_dark_corner'], r['_dark_border']), reverse=True)
chosen=None
for r in rows_sorted:
    img_path = root / r['part'] / (r['image_id'] + '.jpg')
    if img_path.exists() and r['dx'] in ('mel','bkl'):
        chosen=(r,img_path)
        break
if chosen is None:
    for r in rows_sorted:
        img_path = root / r['part'] / (r['image_id'] + '.jpg')
        if img_path.exists():
            chosen=(r,img_path)
            break
if chosen is None:
    raise SystemExit('No image found')

r, img_path = chosen
img = Image.open(img_path).convert('RGB')
orig_out = out_dir / 'HAM10000_dark_border_corner_original.jpg'
ann_out = out_dir / 'HAM10000_dark_border_corner_annotated.jpg'
img.save(orig_out, quality=95)

ann = img.copy()
draw = ImageDraw.Draw(ann)
w,h = ann.size
# border band and corner boxes
border = max(28, int(min(w,h)*0.07))
red=(255,0,0)
yellow=(255,230,0)
black=(0,0,0)
# Draw edge bands as red rectangles outlines
for i in range(5):
    draw.rectangle([i,i,w-1-i,h-1-i], outline=red)
# corner boxes
cw,ch = int(w*0.18), int(h*0.18)
for box in ([0,0,cw,ch], [w-cw,0,w-1,ch], [0,h-ch,cw,h-1], [w-cw,h-ch,w-1,h-1]):
    for i in range(4):
        draw.rectangle([box[0]+i,box[1]+i,box[2]-i,box[3]-i], outline=red)
# label boxes
label = f"Dark border / corner cue\nimage: {r['image_id']}\ndx: {r['dx']}\ndark_border={r['_dark_border']:.3f}\ndark_corner={r['_dark_corner']:.3f}"
box=[20,20,310,125]
draw.rectangle(box, fill=yellow, outline=black, width=2)
draw.multiline_text((30,30), label, fill=black, spacing=4)
# arrow to dark edge
start=(310,120); end=(w-40,h-40)
draw.line([start,end], fill=red, width=5)
draw.line([end,(end[0]-22,end[1]-3)], fill=red, width=5)
draw.line([end,(end[0]-4,end[1]-22)], fill=red, width=5)
ann.save(ann_out, quality=95)

print('IMAGE_ID=' + r['image_id'])
print('DX=' + r['dx'])
print('PART=' + r['part'])
print('SOURCE=' + str(img_path))
print('ORIGINAL_OUT=' + str(orig_out))
print('ANNOTATED_OUT=' + str(ann_out))
print('DARK_BORDER=' + str(r['_dark_border']))
print('DARK_CORNER=' + str(r['_dark_corner']))
