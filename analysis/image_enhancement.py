"""
Phase 3.9.A — Image Enhancement Script
=======================================
PlantDoc ve web validation görsellerini OpenCV ile preprocess ederek
PlantVillage tarzına yaklaştırır.

Pipeline (her görsel için sırayla):
  1. White balance  — gray world assumption
  2. CLAHE          — LAB renk uzayında L kanalına, clip_limit=3.0
  3. Gamma          — gamma=1.1 (hafif aydınlatma düzeltmesi)

Kullanım:
  python analysis/image_enhancement.py
  python analysis/image_enhancement.py --source plantdoc
  python analysis/image_enhancement.py --source web
  python analysis/image_enhancement.py --source both
  python analysis/image_enhancement.py --source both --no-compare
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT            = Path(__file__).resolve().parent.parent
PLANTDOC_SRC    = ROOT / "data" / "plantdoc" / "PlantDoc-Dataset" / "test"
WEB_SRC         = ROOT / "test_images" / "web_validation"
ENHANCED_ROOT   = ROOT / "test_images" / "enhanced"
PLANTDOC_DST    = ENHANCED_ROOT / "plantdoc"
WEB_DST         = ENHANCED_ROOT / "web_validation"
COMPARISON_DIR  = ROOT / "results" / "enhancement_examples"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
              ".JPG", ".JPEG", ".PNG", ".BMP"}

COMPARE_N = 5   # before/after comparison PNG sayısı

# ---------------------------------------------------------------------------
# Enhancement pipeline — her aşama bağımsız fonksiyon
# ---------------------------------------------------------------------------

def white_balance_gray_world(bgr: np.ndarray) -> np.ndarray:
    """Gray world assumption ile beyaz dengesi uygular.

    Her kanalın ortalamasını genel parlaklık ortalamasına eşitler,
    aydınlatma renk kaymasını telafi eder.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_all = (mean_b + mean_g + mean_r) / 3.0

    scale_b = mean_all / (mean_b + 1e-6)
    scale_g = mean_all / (mean_g + 1e-6)
    scale_r = mean_all / (mean_r + 1e-6)

    b = np.clip(b * scale_b, 0, 255)
    g = np.clip(g * scale_g, 0, 255)
    r = np.clip(r * scale_r, 0, 255)

    return cv2.merge([b, g, r]).astype(np.uint8)


def apply_clahe(bgr: np.ndarray, clip_limit: float = 3.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """LAB renk uzayında L kanalına CLAHE uygular.

    Renk tonlarını bozmadan kontrast iyileştirmesi yapar.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_ch = clahe.apply(l_ch)

    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma(bgr: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """Power-law gamma düzeltmesi uygular.

    gamma > 1 → görsel hafifçe koyulaşır (PlantVillage karanlık arka plan).
    gamma < 1 → görsel aydınlanır.
    """
    inv_gamma = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(bgr, lut)


def enhance(bgr: np.ndarray,
            clip_limit: float = 3.0,
            gamma: float = 1.1) -> np.ndarray:
    """Tam enhancement pipeline: white balance → CLAHE → gamma."""
    bgr = white_balance_gray_world(bgr)
    bgr = apply_clahe(bgr, clip_limit=clip_limit)
    bgr = apply_gamma(bgr, gamma=gamma)
    return bgr


# ---------------------------------------------------------------------------
# Dosya yardımcıları
# ---------------------------------------------------------------------------

def collect_images(src_dir: Path) -> List[Path]:
    """src_dir altındaki tüm görsel dosyalarını listeler (özyinelemeli)."""
    return [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTS
    ]


def dest_path(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    """Kaynak → hedef yolu: klasör hiyerarşisini korur."""
    rel = src_file.relative_to(src_root)
    return dst_root / rel


# ---------------------------------------------------------------------------
# Karşılaştırma PNG'si
# ---------------------------------------------------------------------------

def save_comparison(
    pairs: List[Tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
) -> None:
    """İlk N görsel için yan yana before/after karşılaştırma kaydeder.

    pairs: [(original_bgr, enhanced_bgr, label), ...]
    """
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for row, (orig, enh, label) in enumerate(pairs):
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        enh_rgb  = cv2.cvtColor(enh,  cv2.COLOR_BGR2RGB)

        axes[row][0].imshow(orig_rgb)
        axes[row][0].set_title(f"Original\n{label}", fontsize=8)
        axes[row][0].axis("off")

        axes[row][1].imshow(enh_rgb)
        axes[row][1].set_title(f"Enhanced\n{label}", fontsize=8)
        axes[row][1].axis("off")

    fig.suptitle(
        "Image Enhancement — White Balance → CLAHE → Gamma",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Comparison saved: {output_path}")


# ---------------------------------------------------------------------------
# Kaynak işleme
# ---------------------------------------------------------------------------

def process_source(
    src_root: Path,
    dst_root: Path,
    source_label: str,
    compare_n: int = COMPARE_N,
) -> Tuple[int, int]:
    """Bir kaynak klasörü altındaki tüm görselleri enhance eder.

    Returns
    -------
    (n_ok, n_fail)
    """
    images = collect_images(src_root)
    if not images:
        print(f"  [UYARI] Görsel bulunamadı: {src_root}")
        return 0, 0

    print(f"\n{'-'*60}")
    print(f"  Kaynak : {src_root}")
    print(f"  Hedef  : {dst_root}")
    print(f"  Toplam : {len(images)} görsel")
    print(f"{'-'*60}")

    n_ok, n_fail = 0, 0
    comparison_pairs: List[Tuple[np.ndarray, np.ndarray, str]] = []

    for img_path in sorted(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [HATA] Okunamadı: {img_path.name}")
            n_fail += 1
            continue

        enhanced = enhance(bgr)

        out = dest_path(img_path, src_root, dst_root)
        out.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(out), enhanced)
        if ok:
            n_ok += 1
        else:
            print(f"  [HATA] Yazılamadı: {out}")
            n_fail += 1
            continue

        if len(comparison_pairs) < compare_n:
            comparison_pairs.append((bgr, enhanced, img_path.name))

    # Karşılaştırma PNG
    if comparison_pairs:
        cmp_path = COMPARISON_DIR / f"compare_{source_label}.png"
        save_comparison(comparison_pairs, cmp_path)

    print(f"  Başarılı: {n_ok}  |  Hatalı: {n_fail}")
    return n_ok, n_fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhancement pipeline: white balance → CLAHE → gamma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["plantdoc", "web", "both"],
        default="both",
        help="Hangi kaynak işlenecek (default: both)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Karşılaştırma PNG'lerini oluşturma",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global COMPARE_N
    if args.no_compare:
        COMPARE_N = 0

    print("=" * 60)
    print("  Phase 3.9.A — Image Enhancement")
    print(f"  Pipeline : Gray World WB -> CLAHE(clip=3.0) -> Gamma(1.1)")
    print(f"  Kaynak   : {args.source}")
    print("=" * 60)

    total_ok = total_fail = 0

    if args.source in ("plantdoc", "both"):
        if not PLANTDOC_SRC.exists():
            print(f"[HATA] PlantDoc dizini bulunamadı: {PLANTDOC_SRC}")
            if args.source == "plantdoc":
                sys.exit(1)
        else:
            ok, fail = process_source(PLANTDOC_SRC, PLANTDOC_DST, "plantdoc")
            total_ok += ok
            total_fail += fail

    if args.source in ("web", "both"):
        if not WEB_SRC.exists():
            print(f"[HATA] Web validation dizini bulunamadı: {WEB_SRC}")
            if args.source == "web":
                sys.exit(1)
        else:
            ok, fail = process_source(WEB_SRC, WEB_DST, "web_validation")
            total_ok += ok
            total_fail += fail

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"  Toplam başarılı : {total_ok}")
    print(f"  Toplam hatalı   : {total_fail}")
    print(f"  Enhanced hedef  : {ENHANCED_ROOT}")
    print(f"  Karşılaştırma   : {COMPARISON_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
