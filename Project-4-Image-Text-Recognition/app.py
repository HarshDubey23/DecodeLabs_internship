"""
Project 4 - Image / Text Recognition (Improved v2) - OCR Pipeline
DecodeLabs AI Internship

Key fixes over the original:
  1. Confidence gate lowered from 80% → 40%  (real scans rarely hit 80%)
  2. 2× image upscaling BEFORE OCR            (biggest single accuracy boost)
  3. CLAHE contrast enhancement               (fixes uneven lighting in scans)
  4. Bilateral filter instead of Gaussian     (preserves text edges better)
  5. Both Otsu + Adaptive thresholding tried  (best result auto-selected)
  6. PSM modes 3, 6, 11 all tried             (best word-count auto-selected)
  7. --lang flag added                        (use 'por+eng' for Portuguese)
  8. Saves debug_preprocessed.jpg             (shows what Tesseract sees)

Usage:
    python app.py --image input_images/sample1.png
    python app.py --image input_images/sample2.png --lang por+eng --threshold 35
"""

import argparse
import os

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image as PILImage

DEFAULT_CONFIDENCE_THRESHOLD = 40    # was 80 — much more realistic for scanned docs
DEFAULT_LANG = "eng"                 # override with --lang por+eng for Portuguese docs
DEFAULT_SCALE = 2.0                  # upscale factor before OCR


# ──────────────────────────────────────────────────────────────────────
# 1.  UPSCALE — biggest single improvement for low-DPI scans
# ──────────────────────────────────────────────────────────────────────
def upscale(image, scale: float = 2.0):
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )


# ──────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING STRATEGIES
# ──────────────────────────────────────────────────────────────────────
def _enhance(gray):
    """Shared step: bilateral denoise + CLAHE contrast boost."""
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def preprocess_otsu(gray):
    """Otsu global threshold — good for high-contrast clean documents."""
    enhanced = _enhance(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def preprocess_adaptive(gray):
    """Adaptive threshold — better for uneven lighting / aged paper."""
    enhanced = _enhance(gray)
    return cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


# ──────────────────────────────────────────────────────────────────────
# 3.  OCR
# ──────────────────────────────────────────────────────────────────────
def run_ocr(processed_image, lang: str = DEFAULT_LANG, psm: int = 6):
    """Run pytesseract and return per-word data dict including confidence."""
    pil_image = PILImage.fromarray(processed_image)
    config = f"--psm {psm} --oem 3"
    return pytesseract.image_to_data(
        pil_image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )


def count_accepted(ocr_data, threshold: int) -> int:
    count = 0
    for i, word in enumerate(ocr_data["text"]):
        if not word.strip():
            continue
        try:
            conf = int(float(ocr_data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if conf >= threshold:
            count += 1
    return count


def best_ocr_result(gray, lang, threshold):
    """
    Try every combo of {Otsu, Adaptive} × {PSM 6, 3, 11}.
    Returns whichever gives the most words passing the confidence gate.
    """
    combos = [
        ("otsu",     preprocess_otsu,      6),
        ("otsu",     preprocess_otsu,      3),
        ("otsu",     preprocess_otsu,     11),
        ("adaptive", preprocess_adaptive,  6),
        ("adaptive", preprocess_adaptive,  3),
        ("adaptive", preprocess_adaptive, 11),
    ]
    best_data, best_thresh_img, best_count, best_label = None, None, -1, "none"

    for prep_name, prep_fn, psm in combos:
        try:
            thresh_img = prep_fn(gray)
            data = run_ocr(thresh_img, lang=lang, psm=psm)
            count = count_accepted(data, threshold)
            label = f"{prep_name}/PSM-{psm}"
            if count > best_count:
                best_count = count
                best_data = data
                best_thresh_img = thresh_img
                best_label = label
        except Exception:
            continue

    return best_data, best_thresh_img, best_count, best_label


# ──────────────────────────────────────────────────────────────────────
# 4.  ANNOTATE  (bounding boxes drawn on the ORIGINAL-size image)
# ──────────────────────────────────────────────────────────────────────
def filter_and_annotate(original_image, ocr_data, threshold, scale):
    accepted_words = []
    annotated = original_image.copy()

    for i, word in enumerate(ocr_data["text"]):
        word = word.strip()
        if not word:
            continue
        try:
            conf = int(float(ocr_data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1

        if conf >= threshold:
            # Divide box coords by upscale factor to map back to original size
            x = int(ocr_data["left"][i]   / scale)
            y = int(ocr_data["top"][i]    / scale)
            w = int(ocr_data["width"][i]  / scale)
            h = int(ocr_data["height"][i] / scale)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                annotated,
                f"{word}({conf}%)",
                (x, max(y - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 0),
                1,
            )
            accepted_words.append((word, conf))

    return annotated, accepted_words


# ──────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Project 4 - Improved OCR Pipeline v2")
    parser.add_argument("--image",     default="input_images/sample1.jpg",
                        help="Path to input image")
    parser.add_argument("--lang",      default=DEFAULT_LANG,
                        help="Tesseract language code(s), e.g. 'eng', 'por+eng'")
    parser.add_argument("--threshold", type=int, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Confidence gate 0-100 (default: 40)")
    parser.add_argument("--scale",     type=float, default=DEFAULT_SCALE,
                        help="Upscale factor before OCR (default: 2.0)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        print("Add a photo to input_images/ and try again.")
        return

    os.makedirs("output", exist_ok=True)

    # Load → upscale → greyscale
    original = cv2.imread(args.image)
    upscaled = upscale(original, scale=args.scale)
    gray     = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Try all combos, pick the best
    ocr_data, best_thresh_img, accepted_count, best_label = best_ocr_result(
        gray, args.lang, args.threshold
    )

    # Annotate on original-size image
    annotated, accepted_words = filter_and_annotate(
        original, ocr_data, args.threshold, args.scale
    )

    # Save outputs
    cv2.imwrite("output/output_image.jpg",        annotated)
    cv2.imwrite("output/debug_preprocessed.jpg",  best_thresh_img)  # handy for debugging

    extracted_text = " ".join(w for w, _ in accepted_words)
    with open("output/detected_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    # Report
    total_detected = sum(1 for t in ocr_data["text"] if t.strip())
    print("=" * 57)
    print(" DecodeLabs OCR Report — v2 (Improved)")
    print("=" * 57)
    print(f"  Best strategy          : {best_label}")
    print(f"  Language               : {args.lang}")
    print(f"  Upscale factor         : {args.scale}x")
    print(f"  Confidence gate        : {args.threshold}%")
    print(f"  Words found by Tess.   : {total_detected}")
    print(f"  Words past gate        : {accepted_count}")
    if accepted_words:
        avg_conf = sum(c for _, c in accepted_words) / len(accepted_words)
        print(f"  Avg accepted conf.     : {avg_conf:.1f}%")
    print(f"  Annotated image saved  : output/output_image.jpg")
    print(f"  Debug preprocessed     : output/debug_preprocessed.jpg")
    print(f"  Text file saved        : output/detected_text.txt")

    if accepted_words:
        print()
        print("Extracted text:")
        print("-" * 40)
        preview = extracted_text[:500]
        print(preview + ("..." if len(extracted_text) > 500 else ""))
    else:
        print()
        print("Still no text passed the gate. Try these in order:")
        print("  1. Lower threshold  ->  --threshold 25")
        print("  2. Portuguese lang  ->  --lang por+eng")
        print("     (requires: sudo apt install tesseract-ocr-por  on Linux)")
        print("     (Windows: download por.traineddata into Tesseract tessdata/ folder)")
        print("  3. Higher scale     ->  --scale 3.0")
        print("  4. Use a cleaner image (printed book page, receipt, poster)")


if __name__ == "__main__":
    main()