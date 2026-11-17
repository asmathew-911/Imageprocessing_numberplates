"""Run from project root: python self_test.py"""
import tempfile
import zipfile
from pathlib import Path
from fastapi.testclient import TestClient
from prepare_dataset import extract, image_files
from anpr import imread, detect_plates
from reference_mode import reference_entries, run_reference_audit
from app import app


def main():
    imgs=image_files('data')
    assert imgs,'No example images installed.'
    image=imread(imgs[0]); assert image is not None
    results=detect_plates(image)
    # Regression fixture: image pixels only, not XML or filename decoding.
    sample=imread('data/google_images/car-wbs-MH20EE7598_00000.png')
    assert sample is not None
    predicted=detect_plates(sample,max_plates=1)
    assert predicted and predicted[0].text=='MH20EE7598', (predicted[0].text if predicted else 'NOT FOUND')
    print('Independent image-only guesses:',[(r.text,r.status) for r in results])
    with tempfile.TemporaryDirectory() as d:
        z=Path(d)/'tiny.zip'
        with zipfile.ZipFile(z,'w') as f:
            f.writestr('google_images/test.jpg',imgs[0].read_bytes())
            f.writestr('video_images/frame.jpg',imgs[0].read_bytes())
        stats=extract(z,Path(d)/'out')
        assert stats['images']==1 and stats['video_images_excluded']==1
    first_with_xml=next(p for p in imgs if p.with_suffix('.xml').exists())
    refs=reference_entries(first_with_xml)
    assert refs and all(r['source'].startswith('XML') for r in refs)
    c=TestClient(app)
    assert c.get('/api/health').status_code==200
    assert c.get('/api/gallery').status_code==200
    blind=c.post('/api/analyze/0?mode=classical')
    assert blind.status_code==200 and blind.json()['mode']=='classical'
    assert not blind.json()['reference_entries']
    ref=c.post('/api/analyze/0?mode=reference')
    assert ref.status_code==200 and ref.json()['mode']=='reference'
    assert not ref.json()['predictions']
    assert ref.json()['reference_entries']
    with tempfile.TemporaryDirectory() as d:
        r=run_reference_audit('data',d,save_annotated=False)
        assert r['xml_annotation_coverage']==r['processed_images']
    print('PASS: no-ML mode, separate XML mode, ZIP video exclusion, website endpoints')

if __name__=='__main__': main()
