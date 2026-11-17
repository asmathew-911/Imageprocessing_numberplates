"""Verify every eligible photo against the SHA256 + XML reference catalog.

This tests annotation retrieval, source-image integrity, and coordinate validity.
It does NOT establish optical character recognition accuracy from pixels.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from prepare_dataset import image_files
from batch import ground_truth
from anpr import imread
from dataset_catalog import lookup_image_file,load_catalog,normalize,label_warning

FIELDS=('filename','matched_by_hash_and_filename','all_xml_texts_agree',
        'all_xml_boxes_agree','valid_boxes','label','box','needs_manual_review',
        'review_reason','result_source')


def verify_all(data_dir,report_path='results/ALL_442_DATASET_CHECK.csv'):
    paths=image_files(data_dir)
    catalog=load_catalog(); report_path=Path(report_path)
    report_path.parent.mkdir(exist_ok=True,parents=True)
    matched=valid=exact=flagged=0
    rows=[]
    for p in paths:
        item=lookup_image_file(p,catalog)
        gt=ground_truth(p)
        image=imread(p)
        has=bool(item)
        same_text=has and gt is not None and [normalize(t) for t,b in gt]==[normalize(a['text']) for a in item['annotations']]
        same_box=has and gt is not None and [list(b) for t,b in gt]==[a['box'] for a in item['annotations']]
        valid_box=bool(image is not None and has and all(0<=a['box'][0]<a['box'][0]+a['box'][2]<=image.shape[1] and 0<=a['box'][1]<a['box'][1]+a['box'][3]<=image.shape[0] for a in item['annotations']))
        warnings=[a['review_reason'] for a in item['annotations'] if a['review_reason']] if has else ['Not found in SHA256 catalog']
        matched+=int(has);exact+=int(same_text and same_box);valid+=int(valid_box);flagged+=int(bool(warnings))
        rows.append({'filename':p.name,'matched_by_hash_and_filename':int(has),'all_xml_texts_agree':int(bool(same_text)),
            'all_xml_boxes_agree':int(bool(same_box)),'valid_boxes':int(valid_box),
            'label':';'.join(a['text'] for a in item['annotations']) if has else '',
            'box':';'.join(str(a['box']) for a in item['annotations']) if has else '',
            'needs_manual_review':int(bool(warnings)),'review_reason':';'.join(warnings),
            'result_source':'PROVIDED_XML_NOT_IMAGE_OCR'})
    with report_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    result={'images_tested':len(paths),'images_matched_catalog':matched,'xml_text_and_boxes_exact':exact,
            'images_with_valid_boxes':valid,'need_manual_review':flagged,'images_without_catalog_match':len(paths)-matched,
            'source':'SHA256 match + supplied XML (NOT automatic reading)','csv':str(report_path)}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/google_images');p.add_argument('--output',default='results/ALL_442_DATASET_CHECK.csv')
    a=p.parse_args();verify_all(a.data,a.output)
