from PIL import Image, ImageDraw
from pathlib import Path
src = Path(r'E:\小龙虾\synthetic_corner_patch\test_id\cat\test_id_000012.png')
out_dir = Path(r'C:\Users\PeiYuan Hao\.openclaw\workspace-summarize_agent\reports')
out_dir.mkdir(parents=True, exist_ok=True)
orig_out = out_dir / 'synthetic_corner_patch_代表图_original.png'
ann_out = out_dir / 'synthetic_corner_patch_代表图_annotated.png'
img = Image.open(src).convert('RGB')
img.save(orig_out)
scale = 16
big = img.resize((img.width*scale, img.height*scale), Image.Resampling.NEAREST)
draw = ImageDraw.Draw(big)
# patch bbox in original: (28,28)-(31,31), convert to enlarged coords
x1,y1,x2,y2 = 28*scale, 28*scale, (31+1)*scale-1, (31+1)*scale-1
# red rectangle around patch
for i in range(6):
    draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=(255,0,0))
# arrow from label area to patch
start = (300, 180)
end = (x1+10, y1+10)
draw.line([start, end], fill=(255,0,0), width=6)
# simple arrow head
ex, ey = end
for p in [(ex,ey), (ex-18, ey-8), (ex-10, ey-26), (ex,ey), (ex-26, ey-10), (ex-8, ey-18)]:
    pass
# arrow head as two lines
draw.line([(ex,ey),(ex-18,ey-8)], fill=(255,0,0), width=6)
draw.line([(ex,ey),(ex-8,ey-18)], fill=(255,0,0), width=6)
# label box
label_box = [70, 110, 320, 190]
draw.rectangle(label_box, fill=(255,255,0), outline=(180,140,0), width=4)
# text fallback using default font
text = 'Shortcut patch\n(bottom-right)'
draw.multiline_text((85, 125), text, fill=(0,0,0), spacing=8)
ann_out.parent.mkdir(parents=True, exist_ok=True)
big.save(ann_out)
print(orig_out)
print(ann_out)
