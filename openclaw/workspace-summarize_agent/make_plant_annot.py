from pathlib import Path
from PIL import Image, ImageDraw
src = Path(r'E:\小龙虾\plant_disease_qualitative\low_fg_area.jpg')
out_dir = Path(r'C:\Users\PeiYuan Hao\.openclaw\workspace-summarize_agent\reports')
out_dir.mkdir(parents=True, exist_ok=True)
orig = out_dir / 'PlantDisease_low_fg_area_original.jpg'
ann = out_dir / 'PlantDisease_low_fg_area_annotated.jpg'
img = Image.open(src).convert('RGB')
img.save(orig, quality=95)
canvas = img.copy()
draw = ImageDraw.Draw(canvas)
w,h = canvas.size
red=(255,0,0); yellow=(255,230,0); black=(0,0,0)
# highlight border/background zones and foreground crop issue
margin_x=max(20,int(w*0.08)); margin_y=max(20,int(h*0.08))
for i in range(6):
    draw.rectangle([i,i,w-1-i,h-1-i], outline=red)
# inner approximate foreground area rectangle to show crop/leaf coverage cue
inner=[margin_x, margin_y, w-margin_x, h-margin_y]
for i in range(4):
    draw.rectangle([inner[0]+i,inner[1]+i,inner[2]-i,inner[3]-i], outline=(255,80,0))
label = 'Crop / foreground coverage cue\nLow fg_area: more visible background\nRisk: model may learn composition, not disease'
box=[20,20,min(w-20,430),140]
draw.rectangle(box, fill=yellow, outline=black, width=2)
draw.multiline_text((34,34), label, fill=black, spacing=6)
# arrows to border/background
start=(min(w-80,430),135); end=(w-40,h//2)
draw.line([start,end], fill=red, width=6)
draw.line([end,(end[0]-24,end[1]-10)], fill=red, width=6)
draw.line([end,(end[0]-20,end[1]+16)], fill=red, width=6)
canvas.save(ann, quality=95)
print('SOURCE=' + str(src))
print('ORIGINAL=' + str(orig))
print('ANNOTATED=' + str(ann))
print('SIZE=', w, h)
