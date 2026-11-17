"""Run independent non-ML ANPR on all allowed images; evaluate XML separately."""
from __future__ import annotations
import argparse,csv,re,xml.etree.ElementTree as ET
from pathlib import Path
from anpr import imread,imwrite,detect_plates,annotate,_iou
from prepare_dataset import image_files

CSV_FIELDS=['filename','status','predicted_plate','confidence','plates_found',
            'bounding_boxes','ground_truth_for_evaluation_only','best_iou',
            'localized_iou_50','correct_full_text_on_matching_box','annotated_image']


def ground_truth(img_path):
    xml=img_path.with_suffix('.xml')
    if not xml.exists():return None
    try:
        root=ET.parse(xml).getroot()
        annotations=[]
        for obj in root.findall('.//object'):
            b=obj.find('bndbox')
            if b is None:continue
            xmin,ymin,xmax,ymax=[int(float(b.findtext(t,'0'))) for t in ('xmin','ymin','xmax','ymax')]
            annotations.append((obj.findtext('name',''),(xmin,ymin,xmax-xmin,ymax-ymin)))
        return annotations
    except (ET.ParseError,ValueError):return None


def norm(text):return re.sub(r'[^A-Z0-9]','',str(text).upper())


def process_one(path,out_dir=None,max_candidates=20):
    path=Path(path)
    img=imread(path)
    if img is None:
        return dict.fromkeys(CSV_FIELDS,'')|{'filename':str(path),'status':'invalid_image'}
    results=detect_plates(img,max_plates=3,max_candidates=max_candidates)
    gt=ground_truth(path) # Read ONLY AFTER prediction; used solely for metrics.
    best_iou='';localized='';exact='';truth=''
    if gt:
        truth='; '.join(text for text,_ in gt)
        overlaps=[(text,r,_iou(r.box,gt_box)) for text,gt_box in gt for r in results]
        best_iou=round(max((v for _,_,v in overlaps),default=0),3)
        localized=int(all(any(_iou(r.box,bb)>=.50 for r in results) for _,bb in gt))
        exact=int(all(any(_iou(r.box,bb)>=.50 and norm(r.text)==norm(text) for r in results)
                      for text,bb in gt))
    out_path=''
    if out_dir is not None:
        out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
        # Prefix with parent folder to avoid collisions.
        out_path=out_dir/(path.parent.name+'__'+path.name+'.jpg')
        imwrite(out_path,annotate(img,results))
    return {
        'filename':str(path),'status': 'read' if any(r.status=='read' for r in results)
                                        else ('uncertain' if results else 'not_found'),
        'predicted_plate':'; '.join(r.text or 'UNKNOWN' for r in results),
        'confidence':'; '.join(f'{r.confidence:.3f}' for r in results),
        'plates_found':len(results),
        'bounding_boxes':'; '.join(str(r.box) for r in results),
        'ground_truth_for_evaluation_only':truth,
        'best_iou':best_iou,'localized_iou_50':localized,
        'correct_full_text_on_matching_box':exact,
        'annotated_image':str(out_path)
    }


def run_batch(data_dir,output_dir='results',limit=0,progress=None,save_annotated=True,max_candidates=20):
    files=image_files(data_dir)
    if limit:files=files[:limit]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    csvfile=out/'predictions.csv'
    processed=0;annotated=0;localized=0;correct=0
    with open(csvfile,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i,path in enumerate(files):
            row=process_one(path,out/'annotated' if save_annotated else None,
                            max_candidates=max_candidates)
            writer.writerow(row);f.flush()
            processed+=1
            if row['localized_iou_50']!='':
                annotated+=1
                localized+=int(row['localized_iou_50'])
                correct+=int(row['correct_full_text_on_matching_box'])
            if progress:progress(processed,len(files),path.name,annotated,localized,correct)
    return {
        'processed_images':processed,'with_xml_reference':annotated,
        'detected_iou_50':localized,'detector_recall':round(localized/max(annotated,1)*100,2),
        'full_plate_exact_on_iou_50':correct,
        'end_to_end_exact_rate':round(correct/max(annotated,1)*100,2),
        'csv':str(csvfile.resolve())
    }

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',default='data/google_images')
    ap.add_argument('--output',default='results')
    ap.add_argument('--limit',type=int,default=0,help='0 means all images')
    ap.add_argument('--no-annotated-images',action='store_true')
    ap.add_argument('--max-candidates',type=int,default=20)
    args=ap.parse_args()
    def log(i,n,name,a,l,c):
        if i%20==0 or i==n:print(f'Completed {i}/{n} | matched={l}/{a}, exact={c}/{a}',flush=True)
    print(run_batch(args.dataset,args.output,args.limit,log,
                    not args.no_annotated_images,args.max_candidates))
