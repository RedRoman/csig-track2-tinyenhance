from pathlib import Path
import json, hashlib
from PIL import Image, ImageDraw, ImageFont
ROOT=Path('/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/matched_pilot_visual')
OUT=ROOT/'inspection_roi'
VARIANTS=('input','control','conservative','aligned')
ROIS={
 7:[('center',(1792,1280,2304,1792)),('blue_panel',(2112,864,2624,1376)),('blue_detail',(2352,976,2608,1232))],
 16:[('center',(1792,1280,2304,1792)),('left_wall',(384,960,896,1472)),('right_wall',(3264,960,3776,1472))],
 23:[('center',(1792,1280,2304,1792)),('glass_middle',(2048,928,2560,1440)),('glass_lower',(1280,2096,1792,2608))]
}
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',15)
manifest={'coordinate_convention':'Pillow (left,top,right,bottom), right/bottom exclusive; origin top-left; native source pixels',
'originals_unchanged':True,'roi_processing':'crop only; output PNG; no contrast, sharpening, denoising or color edits',
'collage_magnification':'1x for 512px ROI; 2x nearest-neighbor for 256px blue_detail; original 1x ROI separately retained',
'overview_processing':'downsample to 1000px width with LANCZOS for locating ROIs only; scale=1000/4096; labels/boxes outside scientific ROI data',
'variant_order':VARIANTS,'cases':{}}
for case,regions in ROIS.items():
 sourceims={};records={}
 for variant in VARIANTS:
  p=ROOT/'originals'/variant/f'case{case}.{ "jpg" if variant=="input" else "png" }'
  im=Image.open(p).convert('RGB');sourceims[variant]=im
  records[variant]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':im.size}
 assert len({im.size for im in sourceims.values()})==1
 roi_records=[]
 W=4*512+5*12;H=len(regions)*(512+50)+12
 collage=Image.new('RGB',(W,H),(245,245,245));draw=ImageDraw.Draw(collage)
 for row,(name,box) in enumerate(regions):
  roi_record={'name':name,'box':box,'native_size':[box[2]-box[0],box[3]-box[1]],'native_roi_files':{},'display_scale':2 if box[2]-box[0]==256 else 1}
  for col,variant in enumerate(VARIANTS):
   roi=sourceims[variant].crop(box)
   dst=OUT/f'case{case}_{name}_{variant}_native.png';roi.save(dst)
   roi_record['native_roi_files'][variant]=str(dst)
   scale=roi_record['display_scale'];panel=roi.resize((512,512),Image.Resampling.NEAREST) if scale==2 else roi
   x=12+col*(512+12);y=12+row*(512+50)
   draw.text((x,y),f'case{case} / {name} / {variant} / {scale}x',font=font,fill=(0,0,0))
   draw.text((x,y+23),f'ROI {box}',font=small,fill=(30,30,30));collage.paste(panel,(x,y+50))
  roi_records.append(roi_record)
 path=OUT/f'case{case}_roi_collage.png';collage.save(path)
 overview=sourceims['control'].resize((1000,750),Image.Resampling.LANCZOS)
 drawov=ImageDraw.Draw(overview)
 for name,box in regions:
  scaled=tuple(round(v*1000/4096) for v in box);drawov.rectangle(scaled,outline=(255,255,0),width=2)
  drawov.text((scaled[0]+2,scaled[1]+2),name,font=small,fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
 overview.save(OUT/f'case{case}_roi_locations.png')
 manifest['cases'][str(case)]={'source_records':records,'rois':roi_records,'collage':str(path)}
(OUT/'roi_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v['collage'] for k,v in manifest['cases'].items()}))
