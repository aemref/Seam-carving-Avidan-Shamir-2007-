# Görev dosyası — Emre

Bu dosya Emre'nin Claude oturumu içindir. Kullanım: proje klasöründe Claude Code'u aç ve "`gorevler/emre.md` dosyasını oku, Dalga 0'ı yap" de. Emre depo sahibidir; kapıları o açar.

## 1. Payın

| Tür | İş |
|---|---|
| Çekirdek modül | `seamcarving/seam.py`: dinamik programlama ile minimum seam |
| Destek kodu | İskelet ve CI, `seamcarving/types.py`, `seamcarving/resize.py`, `experiments/common.py`, `experiments/run_all.py` |
| Deney | D6 Süre |
| Fotoğraf | manzara, nesne, insan, yogun (birer tane) |
| Rapor | `02_2_seam_dp.md`, `04_implementation_runtime.md` |
| Dalga 3 görevi | `run_all.py`, README, son birleştirme, `v1.0` etiketi |
| Koordinasyon | Kapıları açmak, sözleşme değişikliklerine karar vermek |

## 2. Ortak kurallar

Bu bölüm beş görev dosyasında da aynıdır.

### 2.1 Proje

Avidan ve Shamir'in 2007 tarihli "Seam Carving for Content-Aware Image Resizing" makalesini NumPy ile sıfırdan uyguluyoruz (INF605 final projesi). Beş kişi aynı anda, her biri kendi bilgisayarında, kendi GitHub hesabı ve kendi Claude oturumuyla çalışıyor.

- Algoritma, tuzaklar, ders bağlantıları ve deney planı: `PROJE_HARITASI.md`. Kod yazmadan önce Bölüm 3'ü oku.
- Makale: https://doi.org/10.1145/1276377.1276390
- Harita ile bu dosya çelişirse bu dosya geçerlidir.

Kurulum:

```
git clone https://github.com/aemref/Seam-carving-Avidan-Shamir-2007-.git
cd Seam-carving-Avidan-Shamir-2007-
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

### 2.2 Adımlar ve kapılar

| Adım | İş | Başlama koşulu |
|---|---|---|
| Dalga 0 | Emre iskeleti `main`'e gönderir | — |
| Dalga 1 | Herkes kendi modülünü ve birim testlerini yazar | İskelet `main`'de |
| Dalga 2a | Herkes kendi fotoğraflarını ve maskelerini ekler | Kapı 1: beş Dalga 1 PR'ı `main`'de |
| Dalga 2b | Bileşik fonksiyonlar, entegrasyon testleri, deneyler, şekiller, rapor bölümleri | Kod için Kapı 1. Deneyleri koşmak için Kapı 2a: beş fotoğraf PR'ı `main`'de |
| Dalga 3 | Birleştirme görevleri | Kapı 2: beş Dalga 2b PR'ı `main`'de |

Her adımın sonunda PR aç, 2.10'daki özeti kullanıcına ver ve dur. Kapının açıldığını kullanıcın söyler. O zaman `git checkout main && git pull` yap, yeni dal aç, devam et. Dalga 2a PR'ını açtıktan sonra beklemeden Dalga 2b'nin kod kısmına geçebilirsin; deneyleri Kapı 2a'dan sonra koş.

İskelette her modül, sözleşmedeki imzalarla ve `NotImplementedError` fırlatan gövdelerle hazır gelir. Kendi dosyanın gövdelerini doldurursun; başkasının fonksiyonunu içe aktarmak ilk günden çalışır, çağırmak sahibi birleştirene kadar hata verir.

### 2.3 Dosya sahipliği

Yalnızca sahibi olduğun dosyaları oluştur ve düzenle. Beş oturumun çakışmadan birleşebilmesi buna bağlı.

| Dosya | Sahip |
|---|---|
| `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `.github/`, `PROJE_HARITASI.md`, `gorevler/` | Emre |
| `seamcarving/__init__.py`, `types.py`, `seam.py`, `resize.py` | Emre |
| `experiments/__init__.py`, `common.py`, `run_all.py`, `d6_runtime.py` | Emre |
| `seamcarving/filters.py`, `energy.py` | Ahsen |
| `experiments/d1_energy_operators.py`, `d8_noise.py`; `report/report.md` | Ahsen |
| `seamcarving/carve.py`, `baselines.py`, `cli.py`, `__main__.py` | Mirza |
| `experiments/d2_shrink.py`, `d3_energy_preservation.py`; `tests/test_integration.py` | Mirza |
| `seamcarving/enlarge.py`; `tools/prepare_dataset.py`, `tools/check_dataset.py`; `data/README.md` | Furkan |
| `experiments/d4_enlarge.py`, `d7_limits.py` | Furkan |
| `seamcarving/object_removal.py`, `viz.py`; `tools/mask_painter.py`, `tools/make_demo_gif.py` | Berkay |
| `experiments/d5_object_removal.py`; `slides/slides.md`, `slides/demo.md` | Berkay |
| `tests/test_<modül>.py` | Modülün sahibi |
| `data/images/*_<isim>.jpg`, `data/masks/*_<isim>_*.png`, `data/metadata/<isim>.csv` | Adı geçen kişi |
| `results/figures/dN_*`, `results/tables/dN_*` | Deneyin sahibi |
| `report/sections/*.md` | 2.8'deki tabloda yazan kişi |
| `report/contributions/<isim>.md`, `slides/parts/<isim>.md` | Adı geçen kişi |

Başkasının dosyasında hata bulursan düzeltme. Hatayı gösteren en küçük örneği kullanıcına ver; o sahibine iletir.

### 2.4 Veri gösterimi

| Nesne | Gösterim |
|---|---|
| Görüntü | `np.ndarray`, `(H, W, 3)` BGR ya da `(H, W)` gri, `uint8` |
| Enerji | `(H, W)`, `float64` |
| Dikey seam | `(H,)`, `int64`; `seam[r]` = r. satırda seçilen sütun |
| Maske | `(H, W)`, `bool` |

- İndeksleme `img[r, c]`: önce satır, sonra sütun. `x`/`y` adı kullanma; `r`/`c`, `d_rows`/`d_cols` kullan.
- Yatay seam için ayrı algoritma yazılmaz: girdiler transpoze edilir, dikey fonksiyon çağrılır, çıktılar geri çevrilir.
- Fonksiyonlar girdilerini değiştirmez, yeni dizi döndürür.
- Hesap `float64` ile yapılır; `uint8`'e dönerken `np.clip(np.rint(x), 0, 255).astype(np.uint8)`.

### 2.5 Arayüz sözleşmesi

İmzalar bağlayıcıdır. Değişmesi gerekiyorsa dur ve kullanıcına söyle; Emre karar verir ve herkese duyurur. Sözleşmede olmayan yardımcı fonksiyonları kendi dosyana serbestçe ekleyebilirsin.

```python
# seamcarving/types.py ─ Emre
EnergyFn = Callable[[np.ndarray, tuple], np.ndarray]   # (img, carry) -> (H, W) float64
SeamFn = Callable[[np.ndarray], np.ndarray]            # energy -> (H,) int64

# seamcarving/filters.py ─ Ahsen
def correlate2d(img, kernel, border="reflect") -> np.ndarray
    # img (H, W) -> (H, W) float64. kernel kenarları tek sayı.
    # border: "reflect" (= cv2.BORDER_REFLECT_101) | "edge" (= BORDER_REPLICATE) | "zero"
def gaussian_kernel(sigma) -> np.ndarray               # kare, kenarı tek sayı, toplamı 1

# seamcarving/energy.py ─ Ahsen
METHODS = ("forward", "roberts", "sobel", "scharr", "laplacian")
def compute_energy(img, method="sobel", color="gray", blur_sigma=0.0, norm="l1") -> np.ndarray
    # img (H, W) ya da (H, W, 3) -> (H, W) float64, her değer >= 0
    # color: "gray" | "sum" (kanal kanal hesapla, topla). norm: "l1" | "l2". blur_sigma > 0: önce Gauss
def make_energy_fn(**kwargs) -> EnergyFn               # compute_energy'yi (img, carry) imzasına sarar

# seamcarving/seam.py ─ Emre
def cumulative_map(energy) -> tuple[np.ndarray, np.ndarray]   # M (H, W) float64; back (H, W) int8: -1 / 0 / +1
def find_vertical_seam(energy) -> np.ndarray           # (H,) int64; |seam[r] - seam[r-1]| <= 1; eşitlikte en soldaki
def seam_cost(energy, seam) -> float
def is_valid_seam(seam, width) -> bool
    # energy negatif ve çok büyük değerler içerebilir (maske ağırlıkları); inf içermez

# seamcarving/carve.py ─ Mirza
class ShrinkResult(NamedTuple):
    image: np.ndarray       # (H, W-k[, C]), girdiyle aynı dtype
    carry: tuple            # carry dizilerinin aynı seam'lerle kesilmiş halleri
    removed: np.ndarray     # bool (H, W): ORİJİNAL koordinatlarda silinen pikseller; her satırda tam k True
    order: np.ndarray       # int32 (H, W): pikseli silen seam'in sırası 0..k-1; silinmeyenlerde -1
def remove_vertical_seam(arr, seam) -> np.ndarray      # (H, W[, C]) -> (H, W-1[, C])
def shrink_width(img, k, energy_fn=None, seam_fn=None, carry=()) -> ShrinkResult
def shrink_height(img, k, energy_fn=None, seam_fn=None, carry=()) -> ShrinkResult
    # energy_fn None ise compute_energy(img); seam_fn None ise find_vertical_seam.
    # Her adım: energy_fn(img, carry) -> seam_fn(energy) -> görüntü ve carry dizileri aynı seam ile kesilir.
    # shrink_height girdileri transpoze eder, shrink_width çağırır, çıktıları geri çevirir. energy_fn transpoze
    # görüntü ve transpoze carry ile çağrılır. removed/order orijinal (H, W) düzenindedir; her SÜTUNDA tam k True.

# seamcarving/enlarge.py ─ Furkan
def insert_vertical_seams(img, chosen) -> np.ndarray   # chosen bool (H, W), her satırda tam k True -> (H, W+k[, C])
    # Seçili her pikselin hemen sağına yeni piksel eklenir: seçili piksel ile sağ komşusunun ortalaması
    # (son sütunda sol komşu). Orijinal pikseller değişmez, sırası bozulmaz.
def inserted_mask(chosen) -> np.ndarray                # bool (H, W+k): eklenen piksellerin çıktıdaki yerleri
def enlarge_width(img, k, energy_fn=None, seam_fn=None, max_step=0.5) -> np.ndarray
def enlarge_height(img, k, energy_fn=None, seam_fn=None, max_step=0.5) -> np.ndarray
def amplify_content(img, scale=1.2, energy_fn=None, seam_fn=None) -> np.ndarray   # çıktı girdiyle aynı boyutta

# seamcarving/object_removal.py ─ Berkay
class RemovalResult(NamedTuple):
    image: np.ndarray
    n_seams: int
    direction: str          # "vertical" | "horizontal"
def apply_mask_weights(energy, remove_mask=None, protect_mask=None) -> np.ndarray
def seams_needed(mask) -> tuple[int, int]              # (dikey seam sayısı, yatay seam sayısı)
def remove_object(img, remove_mask, protect_mask=None, restore_size=True, direction="auto",
                  dilate=0, energy_fn=None, seam_fn=None) -> RemovalResult

# seamcarving/baselines.py ─ Mirza
def scale(img, new_h, new_w, interpolation=None) -> np.ndarray   # None: küçültmede INTER_AREA, büyütmede INTER_CUBIC
def center_crop(img, new_h, new_w) -> np.ndarray
def best_crop(img, new_h, new_w, energy=None) -> np.ndarray      # toplam enerjisi en yüksek pencere
def remove_columns(img, k, energy=None) -> np.ndarray            # toplam enerjisi en düşük k sütun

# seamcarving/viz.py ─ Berkay
def energy_to_image(energy, log=False) -> np.ndarray   # uint8 (H, W), min-max; log=True ise önce log(1 + e - e.min())
def draw_seams(img, removed, color=(0, 0, 255)) -> np.ndarray    # removed: bool (H, W) ya da tek seam (H,); BGR kopya
def comparison_grid(images, labels, cols=None, cell_height=320) -> np.ndarray   # BGR uint8; etiketler ASCII olmalı
def save_gif(frames, path, fps=15) -> None             # frames: BGR uint8 görüntü listesi

# seamcarving/resize.py ─ Emre
def resize(img, new_h, new_w, energy_fn=None, seam_fn=None, protect_mask=None, order="width_first") -> np.ndarray

# experiments/common.py ─ Emre
IMAGES_DIR, MASKS_DIR, FIG_DIR, TABLE_DIR              # pathlib.Path
def list_images(category=None, person=None) -> list[Path]
def load_image(path) -> np.ndarray                     # BGR uint8
def load_mask(image_path, kind="remove") -> np.ndarray | None    # bool (H, W); kind: "remove" | "protect"
def save_figure(name, img) -> Path                     # results/figures/<name>.png
def save_table(name, rows, header) -> Path             # results/tables/<name>.csv
```

Komut satırı (Mirza, Dalga 3):

```
python -m seamcarving resize GIRDI CIKTI [--width W | --dw N] [--height H | --dh N] [--energy sobel] [--protect MASKE]
python -m seamcarving remove GIRDI MASKE CIKTI [--protect MASKE] [--no-restore]
python -m seamcarving energy GIRDI CIKTI [--method sobel]
python -m seamcarving seams  GIRDI CIKTI --dw N
```

Deney betikleri: her `experiments/dN_*.py` argümansız bir `main()` içerir, `python -m experiments.dN_...` ile çalışır, rastgelelik için `np.random.default_rng(0)` kullanır, gereken fotoğraf ya da maske yoksa hata vermeden bunu yazıp o kısmı atlar.

### 2.6 Veri seti

Veri, ekibin kendi çektiği fotoğraflardır. Kategori adları dosya adında şu yazımla geçer: `manzara`, `nesne`, `insan`, `mimari`, `yogun`.

| Kişi | manzara | nesne | insan | mimari | yogun |
|---|---|---|---|---|---|
| Emre | 1 | 1 | 1 | – | 1 |
| Ahsen | 1 | 1 | 1 | 1 | – |
| Mirza | 1 | 1 | 1 | 1 | – |
| Furkan | 1 | 1 | – | 1 | 1 |
| Berkay | 1 | 1 | 1 | 1 | – |

- `manzara`: geniş boş alanı olan sahne (gökyüzü, deniz, çim). `nesne`: sade ya da dokulu bir arka plan önünde, kadraj genişliğinin en çok %15'ini kaplayan tek nesne. `insan`: en az bir kişi. `mimari`: düz çizgiler (bina, çit, raf). `yogun`: boş alanı olmayan sahne (kalabalık, kitaplık, yazı).
- Ham dosyalar `data/raw/<isim>/<kategori>/` altına konur; bu klasör git'e girmez. HEIC desteklenmez, önce JPEG'e çevrilir (macOS: `sips -s format jpeg girdi.heic --out cikti.jpg`).
- `python tools/prepare_dataset.py --person <isim>` ham dosyaları `data/images/<kategori>_<nn>_<isim>.jpg` olarak yazar (uzun kenar 800 piksel, EXIF'siz) ve `data/metadata/<isim>.csv` dosyasını günceller.
- `python tools/mask_painter.py data/images/<dosya>.jpg` ile maske boyanır: `data/masks/<dosya-adı>_remove.png` ve `<dosya-adı>_protect.png` (beyaz = işaretli, görüntüyle aynı boyut). `nesne` fotoğrafı için silme maskesi, `insan` fotoğrafı için koruma maskesi (yüz ve gövde) zorunludur.
- İnternetten görsel indirme, yapay görsel üretme. Depo herkese açık: tanınabilen kişilerden izin alınmış olmalı.
- Fotoğraf yoksa kullanıcından iste. Gelene kadar gerçek fotoğraf gerektiren şekilleri üretme, yerine yapay görüntü koyma.

### 2.7 Git

- **Kimlik.** İlk commit'ten önce `git config user.name` ve `git config user.email` çıktısını kullanıcına göster. Bu dosyanın sahibine ait değilse commit atma, kullanıcına sor. E-posta, sahibin GitHub hesabında kayıtlı olmalı; yoksa commit'ler onun hesabına bağlanmaz.
- **Dallar.** `<isim>/dalga-1`, `<isim>/fotograflar`, `<isim>/dalga-2`, `<isim>/dalga-3`. `main`'e doğrudan push yok (Dalga 0 hariç).
- **Commit.** İngilizce, `tür(modül): özet`. Türler: `feat`, `fix`, `test`, `perf`, `docs`, `exp`, `data`, `chore`. Her fonksiyon, testiyle birlikte ayrı commit; adımın sonunda tek büyük commit atma.
- **PR.** Hedef `main`. Açıklamada ne eklendiği ve nasıl test edildiği yazar. Push ve PR için kullanıcından onay al. PR'ı insan birleştirir; "Create a merge commit" seçilir, squash kullanılmaz.
- **Geçmiş.** `--force` yok; commit geçmişini, yazarını ya da tarihini değiştirmek yok.
- **Çakışma.** Çakışma çıktıysa sahibi olmadığın bir dosyaya dokunmuşsundur; o değişikliği geri al.

### 2.8 Rapor ve sunum parçaları

Rapor ve sunum dili İngilizcedir (varsayım; değişirse Emre duyurur).

| Dosya (`report/sections/`) | İçerik | Yazan |
|---|---|---|
| `01_introduction.md` | Problem, yöntemin fikri, ne yaptığımız | Ahsen |
| `02_1_energy.md` | Enerji haritası | Ahsen |
| `02_2_seam_dp.md` | Minimum seam ve dinamik programlama | Emre |
| `02_3_shrinking.md` | Küçültme | Mirza |
| `02_4_enlarging.md` | Büyütme | Furkan |
| `02_5_object_removal.md` | Maske ile nesne silme | Berkay |
| `03_course_connection.md` | Dersle ilişki | Ahsen |
| `04_implementation_runtime.md` | Mimari ve süre (D6) | Emre |
| `05_dataset.md` | Veri seti | Furkan |
| `06_1_results_energy.md` | D1, D8 | Ahsen |
| `06_2_results_shrinking.md` | D2, D3 | Mirza |
| `06_3_results_enlarging.md` | D4 | Furkan |
| `06_4_results_object_removal.md` | D5 | Berkay |
| `07_limitations.md` | D7 | Furkan |
| `08_conclusion.md` | Sonuç | Berkay |

- Yöntem bölümleri 250–400 kelime, sonuç bölümleri deney başına 150–300 kelime ve 1–3 şekil.
- Şekiller `![Caption](FIG/<dosya>.png)` biçiminde yazılır; Ahsen birleştirirken yolu düzeltir.
- Sayılar `results/tables/` içindeki dosyalardan alınır. Çalıştırmadığın bir şeyi yazma; kötü çıkan sonucu da yaz.
- `report/contributions/<isim>.md`: bu kişinin projede ne yaptığını anlatan 3–5 cümle.
- `slides/parts/<isim>.md`: 2–3 slayt. Her slayt bir `## Başlık`, en çok 4 madde, bir şekil ve en çok 80 kelimelik `Notes:` paragrafı. Sunum sırası: Ahsen, Emre, Mirza, Furkan, Berkay.

### 2.9 Kalite kuralları

- Çekirdek algoritma (filtreleme, enerji, DP, seam silme ve ekleme) yalnızca NumPy ile yazılır. `cv2` şunlar için serbesttir: dosya okuma ve yazma, renk dönüşümü, `cv2.resize` kıyası, `cv2.dilate`, `cv2.medianBlur`, pencere ve fare, testlerde referans.
- Hazır seam carving paketi ya da kodu kullanma, kopyalama.
- Yeni bağımlılık ekleme. Kurulu olanlar: numpy, opencv-python, matplotlib, pillow, pytest, ruff.
- Kod, yorum ve docstring İngilizce. Tip ipuçları ve kısa docstring: ne alır, ne döndürür, boyutlar.
- Testler küçük yapay görüntülerle çalışır, `data/` içindeki fotoğraflara bağlı değildir, toplamda 30 saniyeyi geçmez.
- PR açmadan önce `ruff check .` ve `pytest -q` temiz olmalı.

### 2.10 Her PR'dan sonra kullanıcına ver

1. Eklenen ve değişen dosyalar.
2. Test çıktısı: geçen ve kalan sayısı.
3. Sözleşmeden sapma, bilinen eksik, atlanan iş.
4. Kullanıcının yapması gerekenler: PR'ı birleştirmek, fotoğraf koymak, maske boyamak.
5. Kısa anlatım: bu adımda yazılan kodun nasıl çalıştığını, kullanıcının sunumda ya da sözlü soruda kendi cümleleriyle anlatabileceği 5–8 cümleyle özetle.

## 3. Dalga 0 — İskelet

Diğer dört kişi bu adımı bekliyor; hızlı ve eksiksiz olmalı. Çalışma klasörü henüz git deposu değil ve uzak depo boş.

- [ ] `git init -b main`; uzak depo: `https://github.com/aemref/Seam-carving-Avidan-Shamir-2007-.git`
- [ ] `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`, `data/raw/`, `INF605_*.pdf` (ders slaytları depoya girmez)
- [ ] `requirements.txt`: numpy, opencv-python, matplotlib, pillow, pytest, ruff
- [ ] `pyproject.toml`: paket adı `seamcarving`, Python 3.10+, pytest ayarı, ruff satır uzunluğu 100
- [ ] `seamcarving/__init__.py` (yalnızca sürüm) ve `seamcarving/types.py`
- [ ] 2.5'teki her modül için taslak dosya: imzalar ve docstring'ler sözleşmeyle birebir, gövdeler `raise NotImplementedError`. `ShrinkResult` ve `RemovalResult` taslakta gerçek `NamedTuple` olarak tanımlı olsun. `cli.py` ve `__main__.py` de taslak olarak gelsin
- [ ] `experiments/__init__.py`; boş klasörler için `.gitkeep`: `data/images`, `data/masks`, `data/metadata`, `results/figures`, `results/tables`, `report/sections`, `report/contributions`, `slides/parts`, `tools`
- [ ] `tests/test_skeleton.py`: bütün modüllerin içe aktarılabildiğini doğrulayan tek test (CI'ın koşacak en az bir testi olsun)
- [ ] `.github/workflows/ci.yml`: Python 3.12, bağımlılıkları kur, `ruff check .`, `pytest -q`
- [ ] `README.md` taslağı: iki cümlelik tanım, kurulum, ekip tablosu (beş isim, GitHub adı, modül), haritaya ve `gorevler/` klasörüne bağlantı
- [ ] `PROJE_HARITASI.md` ve `gorevler/*.md` depoda
- [ ] Yerelde `pip install -e .`, `ruff check .`, `pytest -q` temiz
- [ ] Kullanıcının onayıyla ilk commit ve `main`'e push. Sonra ekibe haber: "İskelet hazır, Dalga 1 başlayabilir"

## 4. Dalga 1 — `seam.py` ve `experiments/common.py`

Dal: `emre/dalga-1`.

### 4.1 `seamcarving/seam.py`

```
M(i, j) = e(i, j) + min( M(i−1, j−1), M(i−1, j), M(i−1, j+1) )     ilk satırda M = e
```

- [ ] `cumulative_map`: satır içi işlem vektörleştirilmiş; Python döngüsü yalnızca satırlar üzerinde. İlk ve son sütunda eksik komşu `+inf` sayılır. `back`, üst satırda gelinen sütunun farkını tutar (−1, 0, +1); ilk satır 0
- [ ] `find_vertical_seam`: son satırın en küçüğünden yukarı geri izleme. Eşitlikte en soldaki: hem son satırda hem üç komşu arasında
- [ ] `seam_cost`, `is_valid_seam`
- [ ] Tuzak: `back` `int8` ise sütun indeksine eklerken `int(...)` ile çevir; NumPy 2'de `167 + np.int8(1)` taşma hatası verir
- [ ] Tuzak: enerji negatif ve çok büyük değerler içerebilir (maske ağırlıkları). Negatif olmama varsayımı yapma

`tests/test_seam.py`:

- [ ] Kaba kuvvet: 200 rastgele harita (kenarlar 2–6, tamsayı enerjiler), tüm geçerli seam'ler denenir, en küçük maliyet DP sonucuna eşit
- [ ] Dönen seam `int64`, uzunluğu H, `is_valid_seam` doğru; `seam_cost(e, seam)` = `M[-1].min()`
- [ ] Sabit enerji haritasında seam sütun 0'dan geçer (eşitlik kuralı)
- [ ] Haritaya sabit eklemek ya da çıkarmak seam'i değiştirmez; −1e7'lik bir bölge varsa seam oradan geçer
- [ ] Kenar durumları: W = 1, W = 2, H = 1
- [ ] Elle hesaplanmış 4 × 4 örnek; girdi dizisi değişmiyor
- [ ] Yerelde ölç (teste koyma): 533 × 800 rastgele haritada tek seam 50 ms'nin altında

### 4.2 `experiments/common.py`

- [ ] Sözleşmedeki yollar ve fonksiyonlar. `list_images`, `<kategori>_<nn>_<isim>.jpg` adlarını ayrıştırır ve ada göre sıralı döndürür; klasör boşsa boş liste
- [ ] `load_mask`: 127 üstü `True`; dosya yoksa `None`; boyut görüntüyle uyuşmuyorsa `ValueError`
- [ ] `tests/test_common.py`: geçici klasörde üretilmiş küçük dosyalarla

## 5. Kapı 1'i aç

- [ ] Beş Dalga 1 PR'ı birleşti mi: `gh pr list --state merged`
- [ ] Güncel `main`'de `pytest -q` yeşil
- [ ] Ekibe haber: "Kapı 1 açık"

## 6. Dalga 2a — Fotoğraflar

Dal: `emre/fotograflar`. İnsan işi: 4 fotoğraf (manzara, nesne, insan, yogun), `nesne` için silme maskesi, `insan` için koruma maskesi. Adımlar 2.6'da. Beş fotoğraf PR'ı birleşince ekibe "Kapı 2a açık" de.

## 7. Dalga 2b — `resize.py`, D6, rapor

Dal: `emre/dalga-2`.

### 7.1 `seamcarving/resize.py`

- [ ] Hedef genişlik küçükse `shrink_width`, büyükse `enlarge_width`; yükseklik için aynısı. `order`: `"width_first"` ya da `"height_first"`
- [ ] `protect_mask` yalnızca küçültmede uygulanır: `carry=(protect_mask,)` ve `apply_mask_weights` ile sarılmış `energy_fn`. İlk yöndeki kesilmiş maske ikinci yöne aktarılır. Büyütmede yok sayıldığı docstring'de yazar
- [ ] Geçersiz hedef boyutta `ValueError`; hedef boyut aynıysa kopya döner
- [ ] `tests/test_resize.py`: dört yön kombinasyonu için çıktı boyutu; aynı boyut; koruma maskeli küçültmede, benzersiz değerli korunan bloğun bütün pikselleri çıktıda

### 7.2 D6 — `experiments/d6_runtime.py`

Soru: süre görüntü boyutuyla nasıl büyüyor ve vektörleştirme ne kazandırıyor?

- [ ] Uzun kenarı 300, 600, 800 ve 1200 piksel olan dört boyut (veri setinden bir fotoğrafın `cv2.resize` ile yeniden boyutlandırılmış halleri). Seam başına süre ve adım dağılımı: enerji, DP, silme
- [ ] İç içe iki Python döngüsüyle yazılmış saf DP (yalnızca bu betiğin içinde) ile vektörleştirilmiş DP kıyası, iki boyutta
- [ ] Genişliği %25 daraltmanın toplam süresi; piksel sayısına karşı süre grafiği ve doğrusal uyum. Makale §5: yöntem piksel sayısında doğrusal; 400 × 500 → 100 × 100 yaklaşık 2.2 sn (2007 donanımı)
- [ ] Çıktı: `results/figures/d6_runtime.png`, `results/tables/d6_runtime.csv`; tabloda işlemci, Python ve NumPy sürümü

### 7.3 Rapor ve sunum

- [ ] `02_2_seam_dp.md`: seam tanımı (Denk. 2–4), 8-bağlantılı yol (ders Hafta 3), DP bağıntısı ve geri izleme, seam başına O(H·W) karmaşıklık, açgözlü seçimin neden optimal olmadığını gösteren küçük sayısal karşı örnek
- [ ] `04_implementation_runtime.md`: modül yapısı, `energy_fn` ve `carry` tasarımı, vektörleştirme, D6 sonuçları
- [ ] `report/contributions/emre.md`, `slides/parts/emre.md` (DP ve süre)

Beş Dalga 2b PR'ı birleşince, `main`'de testler yeşilse ekibe "Kapı 2 açık" de.

## 8. Dalga 3 — Son birleştirme

Dal: `emre/dalga-3`.

- [ ] `experiments/run_all.py`: D1–D8 `main()` fonksiyonlarını sırayla çağırır; `--only d2` seçeneği; her deneyin süresini ve başarısını yazar
- [ ] README son hali: kurulum, Python arayüzü ve komut satırı için hızlı başlangıç, `results/figures/` içinden 6 şekillik sonuç galerisi, ekip ve katkı tablosu (`report/contributions/`), sonuçları yeniden üretme, veri seti notu, kaynakça
- [ ] Diğer dört Dalga 3 PR'ı birleştikten sonra: temiz bir klonda kurulum, `pytest -q`, `python -m experiments.run_all`, README'deki komutların denenmesi
- [ ] Kullanıcının onayıyla `v1.0` etiketi

## 9. İnsan işleri (Emre)

- Dalga 0 push onayı; kapı duyuruları; sözleşme değişikliği kararları.
- 4 fotoğraf çekmek ya da seçmek, iki maskeyi boyamak.
- Kendi PR'larını birleştirmek; inceleme eşi Mirza'nın PR'larına bakmak.
- Her adımın sonundaki kısa anlatımı okuyup `seam.py`'yi kendi cümleleriyle anlatabilmek.
