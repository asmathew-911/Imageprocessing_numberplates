"""Print classical OCR alternatives for each plate character (does NOT read XML).

Usage: python diagnose_one.py "C:\\path\\to\\car.png"
Saves one annotated result under results/.
"""
import argparse
from pathlib import Path
from anpr import imread, imwrite, detect_plates, recognize_plate, annotate, _font_paths, _templates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',help='Path to any JPG, JPEG or PNG image')
    args=parser.parse_args()
    path=Path(args.image)
    im=imread(path)
    if im is None:raise SystemExit('Cannot read image: '+str(path))
    results=detect_plates(im,max_plates=3)
    print('This is image-only recognition. No XML, filename, or ML is used.')
    print('Available local font files:',len(_font_paths()))
    print('Generated reference glyphs:',len(_templates()[1]))
    if not results:print('No plate region detected.');return
    for k,result in enumerate(results,1):
        print('\nPlate',k,'box',result.box,'text',result.text,
              'heuristic confidence',result.confidence)
        _,_,meta=recognize_plate(result.crop)
        for item in meta.get('characters',[]):
            a=', '.join(x['char']+'='+f"{x['similarity']:.3f}" for x in item['alternatives'])
            print(f"  {item['position']:>2}. {item['predicted'] or '?'}: "+a+
                  f" | enclosed holes={item['enclosed_holes']}")
    dest=Path(__file__).resolve().parent/'results'/('diagnosed_'+path.stem+'.jpg')
    imwrite(dest,annotate(im,results))
    print('\nSaved annotated image:',dest)

if __name__=='__main__':main()
