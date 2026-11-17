"""Dataset annotation viewer/audit, NOT automatic plate detection or OCR.

This module deliberately reads dataset XML ground-truth labels. It is intended for
reference visualization and annotation coverage checks, never as evidence of
accuracy from pixels. Independent image-based inference lives in anpr.py.
"""
from __future__ import annotations
import csv
from pathlib import Path
import cv2
from batch import ground_truth
from anpr import imread, imwrite
from prepare_dataset import image_files

FIELDS = ['filename','annotation_available','reference_text_from_xml_NOT_OCR',
          'bounding_box_from_xml_NOT_detection','valid_box','review_needed',
          'review_reason','output_image']


def assess_text(text: str) -> str:
    if not text or not text.strip():
        return 'Missing annotation text'
    # Strings without any digit cannot be confirmed registration numbers.
    # This is a review flag, not a claim that every string with digits is correct.
    if not any(ch.isdigit() for ch in text):
        return 'No digits in XML label; may be vehicle name, description, or unreadable plate'
    return ''


def annotate_xml(image, annotations):
    """Draw an unobtrusive box only. Data provenance remains in the result UI.

    This function visualizes supplied annotations and MUST NOT be reported as
    automatic image-based detection or text recognition.
    """
    out = image.copy()
    h, w = image.shape[:2]
    for _label, (x, y, bw, bh) in annotations:
        x, y, bw, bh = map(int, (x, y, bw, bh))
        cv2.rectangle(out, (x, y), (x+bw, y+bh), (38, 190, 240),
                      max(2, round(min(h, w)/290)))
    return out


def reference_entries(image_path, image=None):
    path=Path(image_path)
    annotations=ground_truth(path) or []
    if image is None:
        image=imread(path)
    if image is None:raise ValueError(f'Unreadable image: {path}')
    h,w=image.shape[:2]
    out=[]
    for label,box in annotations:
        x,y,bw,bh=map(int,box)
        valid=0<=x<x+bw<=w and 0<=y<y+bh<=h
        reason=assess_text(label)
        if not valid:reason = (reason+'; ' if reason else '')+'Bounding box outside image'
        out.append(dict(text=label, box=[x,y,bw,bh], valid_box=valid,
                        review_needed=bool(reason), review_reason=reason,
                        source='XML annotation — NOT recognized from pixels'))
    return out


def run_reference_audit(data_dir,output_dir='results',progress=None,save_annotated=True):
    images=image_files(data_dir)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'reference_audit.csv'
    count=0;with_xml=0;covered=0;boxes=0;review=0
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for idx,path in enumerate(images,1):
            count+=1
            image=imread(path)
            if image is None:
                items=[];valid=False
            else:
                items=reference_entries(path,image)
                valid=bool(items) and all(t['valid_box'] for t in items)
            has=bool(items)
            with_xml+=int(has)
            covered+=int(has and valid)
            boxes+=sum(1 for item in items if item['valid_box'])
            flagged=not has or any(item['review_needed'] for item in items)
            review+=int(flagged)
            output=''
            if save_annotated and image is not None and has:
                output=out/'xml_annotated'/(path.parent.name+'__'+path.name+'.jpg')
                imwrite(output,annotate_xml(image,ground_truth(path) or []))
            writer.writerow({
                'filename':str(path),
                'annotation_available':int(has),
                'reference_text_from_xml_NOT_OCR':'; '.join(i['text'] for i in items),
                'bounding_box_from_xml_NOT_detection':'; '.join(str(i['box']) for i in items),
                'valid_box':int(valid),
                'review_needed':int(flagged),
                'review_reason':'; '.join(i['review_reason'] for i in items if i['review_reason']) or ('Missing XML annotation' if not has else ''),
                'output_image':str(output)
            })
            f.flush()
            if progress:progress(idx,len(images),path.name,with_xml,covered,review)
    return {
        'mode':'reference',
        'processed_images':count,
        'xml_annotation_coverage':with_xml,
        'xml_coverage_percent':round(100*with_xml/max(count,1),2),
        'images_with_valid_xml_box':covered,
        'needs_manual_review':review,
        'source':'Dataset XML labels (not detected or read automatically)',
        'csv':str(csv_path.resolve())
    }

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description='Audit dataset XML labels (NOT OCR).')
    ap.add_argument('--dataset',default='data/google_images')
    ap.add_argument('--output',default='results')
    ap.add_argument('--no-annotated-images',action='store_true')
    args=ap.parse_args()
    print(run_reference_audit(args.dataset,args.output,
                              save_annotated=not args.no_annotated_images))
