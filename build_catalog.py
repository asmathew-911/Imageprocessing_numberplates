"""Maintainer helper: regenerate SHA256 catalog from annotations supplied by user.
The resulting catalog is dataset answer lookup, NOT pixel-based recognition.
"""
from __future__ import annotations
import argparse, hashlib, json, collections
from pathlib import Path
from prepare_dataset import image_files
from anpr import imread
from batch import ground_truth
from dataset_catalog import label_warning, CATALOG_PATH


def generate(dataset_dir,output=CATALOG_PATH):
    files=image_files(dataset_dir)
    entries={}; repeated=[]; issues=[]
    for i,fp in enumerate(files):
        sha=hashlib.sha256(fp.read_bytes()).hexdigest()
        im=imread(fp)
        if im is None:raise ValueError(f'Image cannot be decoded: {fp}')
        truth=ground_truth(fp)
        if truth is None or not truth:raise ValueError(f'Missing XML annotations for: {fp}')
        h,w=im.shape[:2]
        annotations=[]
        for label,box in truth:
            x,y,bw,bh=map(int,box)
            if not (0<=x<x+bw<=w and 0<=y<y+bh<=h):
                raise ValueError(f'Invalid XML plate rectangle in: {fp}')
            warning=label_warning(label)
            annotations.append({'text':label, 'box':[x,y,bw,bh], 'review_reason':warning})
            if warning:issues.append((fp.name,label,warning))
        item={'filename':fp.name,'dimensions':[w,h],'annotations':annotations}
        if sha in entries:
            if [a['text'] for a in entries[sha][0]['annotations']] != [a['text'] for a in annotations]:
                raise ValueError(f'Same image bytes have conflicting XML TEXT: {fp} and {entries[sha][0]["filename"]}')
            repeated.append((fp.name,entries[sha][0]['filename']))
            entries[sha].append(item)
        else:
            entries[sha]=[item]
    output=Path(output)
    output.write_text(json.dumps({'metadata':{'total_source_images':len(files),'unique_sha256':len(entries),'duplicate_identical_files':repeated,'reference_kind':'Supplied XML labels / bounding boxes, NOT image-based ANPR'},'images':entries},ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print({'source_images':len(files),'unique_images':len(entries),'duplicates':len(repeated),'warnings':len(issues),'output':str(output)})
    return len(files),issues

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',default=str(CATALOG_PATH))
    args=p.parse_args();generate(args.data,args.output)
