"""Scientific native-pixel comparisons; originals must match full manifests."""
from pathlib import Path
from PIL import Image, ImageDraw
import hashlib
import html
import json
import os

L = Path('/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search')
OUT = L / 'nr_push5_visuals_20261014'
CASES = [7,16,41,83]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    prior = L/'nr_push4_visuals_20261014'
    old = read(prior/'preview_manifest.json')
    boxes = {case:next(x['box_xyxy_exclusive'] for x in old['previews']
                      if x['type']=='native_roi' and x['case']==case) for case in CASES}
    manifests = {
        'hypir':read(prior/'provenance/push4_a7750_compiled_all/render_manifest.json'),
        'reference':read(L/'nr_push5_20261014/reference/render_manifest.json'),
        'student':read(L/'nr_push5_20261014/student/render_manifest.json'),
    }
    names = [('hypir','HYPIR A7750 / previous model'),
             ('reference','Supplied G4 / reference reproduction'),
             ('student','New student / 250 distillation updates')]
    (OUT/'hypir').mkdir(parents=True,exist_ok=True)
    records = []
    sections = []
    for case in CASES:
        box = boxes[case]
        width,height = box[2]-box[0],box[3]-box[1]
        panel = Image.new('RGB',(3*(width+12)+12,height+48),'#edf1f6')
        draw = ImageDraw.Draw(panel)
        cells = []
        for column,(key,title) in enumerate(names):
            path = OUT/key/f'case{case}.png'
            if key=='hypir' and not path.exists():
                os.link(prior/f'push4_a7750_compiled_all/render/test/case{case}.png',path)
            row = next(r for r in manifests[key]['records'] if r['split']=='test' and r['case']==case)
            digest = sha(path)
            assert digest==row['sha256']
            with Image.open(path) as im:
                assert im.mode=='RGB' and im.size==tuple(row['size'])
                region = im.crop(box)
                x = 12+column*(width+12)
                panel.paste(region,(x,36))
                draw.text((x,12),title,fill='black')
            records.append({'case':case,'model':key,'source':str(path),'source_sha256':digest,
                            'manifest_output':row['output'],'native_size_wh':row['size'],
                            'ROI_xyxy_exclusive':box,'scale':1,'filter':None})
            cells.append(f'<figure><figcaption>{html.escape(title)}</figcaption>'
                         f'<a href="{key}/case{case}.png"><img loading="lazy" src="{key}/case{case}.png"></a>'
                         f'<a href="{key}/case{case}.png">原尺寸PNG</a></figure>')
        destination = OUT/f'roi_case{case}_native.png'
        panel.save(destination)
        for column,(key,_) in enumerate(names):
            with Image.open(OUT/key/f'case{case}.png') as original,Image.open(destination) as rendered:
                x = 12+column*(width+12)
                assert original.crop(box).tobytes()==rendered.crop((x,36,x+width,36+height)).tobytes()
        sections.append(f'<section><h2>case{case}</h2><div class="grid">'+''.join(cells)+
                        f'</div><p>固定ROI：{box}，下图为1:1原像素拼接；点击按原尺寸查看。</p>'
                        f'<a href="{destination.name}"><img class="roi" src="{destination.name}"></a></section>')
    text = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Push5 原始输出比较</title>
<style>body{max-width:1700px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;background:#f5f7fa;color:#172033}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%;height:auto}section{margin:40px 0}
figcaption{font-weight:bold}a{color:#1758a7}.roi{width:100%}</style>
<h1>HYPIR、工具参考模型与本轮新学生</h1>
<p>参考G4来自用户评分工具包；新学生从其权重初始化，向HYPIR A7750的100张测试输出蒸馏250步。验证GT未参与训练。
三列均为实际完整评测的原PNG。ROI沿用上一轮坐标，原图不锐化、不润色。高NR不代表每处结构都更真实；请同时查看完整报告。</p>'''
    (OUT/'index.html').write_text(text+''.join(sections)+'</html>')
    proof = {'status':'PASS','cases':CASES,'source_records':records,
             'all_sources_match_final_full_manifest':True,'all12_ROI_pixel_equal_to_original_crop':True,
             'fixed_prior_ROI_manifest_sha256':sha(prior/'preview_manifest.json'),
             'builder_sha256':sha(Path(__file__)),'no_original_image_modified':True}
    (OUT/'visual_provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','gallery':str(OUT/'index.html'),'source_images':len(records)},ensure_ascii=False))


if __name__=='__main__':
    main()
