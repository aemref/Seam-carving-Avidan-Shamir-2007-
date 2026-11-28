## 5. Dataset

All 20 images are photographs taken by the five team members. No image was downloaded from the internet or generated; unit tests use small synthetic arrays and do not depend on these photos.

### 5.1 Categories

The categories cover both cases where seam carving should work well and cases where the paper reports failures (§6).

| Category (file prefix) | Content | Why it is in the dataset | Used in |
|---|---|---|---|
| `manzara` (landscape) | Large empty area: sky, sea, grass | Best case: many low-energy seams | D1, D2, D3, D4, D8 |
| `nesne` (object) | One object, at most 15% of the frame width | Object removal with a remove mask | D5 |
| `insan` (people) | At least one person | Faces distort; protect-mask demonstration | D1, D2, D7 |
| `mimari` (architecture) | Straight lines: buildings, fences | Straight lines break when seams cross them | D1, D2, D3, D4, D7 |
| `yogun` (dense) | No empty area | Known failure case | D7 |

D6 (run time) uses one photo of any category.

### 5.2 Counts

| Person | manzara | nesne | insan | mimari | yogun | Total |
|---|---|---|---|---|---|---|
| Emre | 1 | 1 | 1 | – | 1 | 4 |
| Ahsen | 1 | 1 | 1 | 1 | – | 4 |
| Mirza | 1 | 1 | 1 | 1 | – | 4 |
| Furkan | 1 | 1 | – | 1 | 1 | 4 |
| Berkay | 1 | 1 | 1 | 1 | – | 4 |
| **Total** | 5 | 5 | 4 | 4 | 2 | 20 |

![All 20 photos with their file names](FIG/dataset_overview.png)

Fifteen photos are 800×600 landscape frames; four are portrait (600×800 three times, 450×800 once) and one is 800×557.

### 5.3 Preparation and naming

Raw files are never committed (`data/raw/<person>/<category>/` is git-ignored). `tools/prepare_dataset.py --person <name>` reduces each file with `cv2.INTER_AREA` so that its long side is 800 pixels and writes it as JPEG (quality 92) to `data/images/<category>_<nn>_<person>.jpg`, for example `manzara_01_furkan.jpg`. It records the original and working size in `data/metadata/<person>.csv` and skips files it has already processed. The originals ranged from 900×1600 to 5120×3840. An 800-pixel long side keeps the NumPy implementation fast enough for the experiments (Section 4) while leaving enough detail to judge distortions.

### 5.4 Masks

Masks are stored as `data/masks/<image-stem>_remove.png` and `_protect.png`: white marks a pixel, and the mask has the size of the image. All five `nesne` photos have a remove mask (one also has a protect mask), and all four `insan` photos have a protect mask covering face and body: ten masks in total. Masks are painted with `tools/mask_painter.py` (left button: remove, right button: protect); the remove mask of `nesne_01_furkan` was instead derived from the object's colour and checked by eye on an overlay.

### 5.5 Privacy

The repository is public, so recognisable people may appear only with their consent (team rule). Re-encoding with OpenCV drops all EXIF metadata, including GPS location; a check of all 20 stored images found no EXIF data. Full-resolution originals are never committed.
