# Character error analysis: MH20EE7598

Test still image: `car-wbs-MH20EE7598_00000.png` from the user's dataset.
This analysis uses the photographic pixels and plate region found by the independent detector, **not** the image filename or XML to decode text.

## Previous issue

- Plate was located, but the detected fourth character (`0`) was predicted as `8`.
- The initial template correlator assigned similar scores to `8` (~0.680) and `0` (~0.659) for one binary threshold variant.
- Correlation alone rewarded the similar dark-and-light stroke arrangements; it did not verify the character's internal holes.
- Some glyphs, notably `E` and `I`, vary with font weight, compression and cropping, so one fixed rendering style is not enough.

## v3 correction

1. Extract each character from grayscale, CLAHE, Otsu/adaptive thresholding and connected components.
2. Create on-demand synthetic templates of digits/letters with OpenCV fonts and **available** Windows/Linux fonts (no font files are redistributed).
3. For each font, render narrow/regular/wide widths and thin/medium/bold strokes.
4. Count **enclosed white/black counters** (holes) from character contours. Clear `0` should have one enclosed hole; conventional `8` has two. Use this as a soft shape-consistency penalty rather than an inflexible rule.
5. Combine classical template matching with letter/digit positions in common Indian registration patterns.
6. Show top 3 alternative character scores and hole counts in the website under **Why these characters?**.

On this test image, the independent result is `MH20EE7598`. In the selected character segmentation the fourth glyph has **one hole**, so the corrected algorithm favors `0`.

## Test scope

- 45 sampled labeled registration plate **crops** (plate coordinates supplied for evaluation in this isolated OCR test): previous 9/45 exact reads; v3 18/45 exact reads.
- This is **not** plate detection accuracy, because the crop test deliberately provides the plate location.
- Full end-to-end recognition on arbitrary photos is less reliable, and we cannot guarantee every font, every plate, or 100% accuracy without machine learning.

## Optional fonts

Add legally obtained `.ttf` or `.otf` font files to `custom_fonts/`, then restart `python app.py`. All such fonts are converted into uppercase/digit templates in memory, not trained on the XML labels. For best results, use fonts like those actually printed on your plate images. No font files are shipped with the project.
