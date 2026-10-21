"""
s01b_prepare_real_data.py

Prepares the real (non-synthetic) rich pictures so they can be run through
s02_detect_picture_elements.py and evaluated with s03_evaluatedetections.py.

For every sub-folder of 01_real_data/ (detected automatically):
  1. Renders the FIRST page of every PDF to a PNG named after the file-name
     prefix:   01_real_data/<folder>/1001_123456.pdf  ->  01_real_data/<folder>/1001.png
  2. Reads GroundTruth.xlsx (in that sub-folder, or one shared file in
     01_real_data/) and writes one ground-truth JSON per "Document name prefix",
     in the same format as the synthetic data:
         01_real_data/<folder>/1001_gt.json
         [{"represents": "Mountain", "image_path": "./01_real_data/<folder>/1001.png",
           "bounding_box": [x1, y1, x2, y2], "element_type": "Icon"}, ...]
     (The "_gt.json" suffix is what s03_evaluatedetections.py expects: it finds
     the image by replacing "_gt.json" with ".png".)
  3. Logs every PDF without ground-truth rows and every ground-truth prefix
     without a PDF to 01_real_data/<folder>/prepare_real_data.log
  4. Deletes PNGs that have no ground truth, and ground-truth JSONs that have
     no PDF/PNG.

GroundTruth.xlsx columns (header row required):
    Document group | Document name prefix | Code | Beginning | End
    (Document group is ignored)
    Beginning / End look like "1: 30|385"  ->  page 1, x = 30, y = 385.
    These are MAXQDA PDF area codings: PDF points with the origin at the
    BOTTOM-left of the page; Beginning = lower-left, End = upper-right corner.
    They are converted to top-left-origin pixel boxes [x1, y1, x2, y2] of the PNG.

Usage:
    python s01b_prepare_real_data.py
    python s01b_prepare_real_data.py --dpi 200 --coord-order yx --no-delete

Requires: pymupdf, pandas, openpyxl, pillow
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from glob import glob

import pandas as pd
import pymupdf  # pip install pymupdf
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Settings (all can be overridden from the command line)
# ----------------------------------------------------------------------------
REAL_DATA_ROOT = "01_real_data"
GT_FILE_NAME = "GroundTruth.xlsx"
LOG_FILE_NAME = "prepare_real_data.log"
CHECK_IMAGE_ROOT = "01_real_data_gt_check"  # overlay images to eyeball the boxes

RENDER_DPI = 300
# Coordinate system of the numbers in Beginning/End:
#   "maxqda_pdf" -> MAXQDA area coding in a PDF: native PDF user space
#                   (points = 1/72 inch, origin BOTTOM-left, y pointing up).
#                   Beginning = lower-left corner, End = upper-right corner.
#                   Page CropBox/MediaBox offsets and /Rotate are handled.
#   "topleft_points" -> points with origin top-left of the page as displayed.
#   "pixels"     -> already pixels of the rendered PNG; used as-is.
COORD_UNIT = "maxqda_pdf"
# Order of the two numbers in "1: 30|385": "xy" -> x=30, y=385 ; "yx" -> y=30, x=385
COORD_ORDER = "xy"

COL_PREFIX = "Document name prefix"
COL_CODE = "Code"
COL_BEGIN = "Beginning"
COL_END = "End"

POSITION_RE = re.compile(r"^\s*(\d+)\s*:\s*(-?\d+(?:\.\d+)?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*$")


# ----------------------------------------------------------------------------
# Logging: one log file per sub-folder
# ----------------------------------------------------------------------------
class FolderLogger:
    def __init__(self):
        self.lines = defaultdict(list)

    def log(self, folder, level, msg):
        line = f"[{level}] {msg}"
        self.lines[folder].append(line)
        print(f"{os.path.basename(folder)}: {line}")

    def write(self):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for folder, lines in self.lines.items():
            path = os.path.join(folder, LOG_FILE_NAME)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"===== Run {stamp} =====\n")
                f.write("\n".join(lines) + "\n\n")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def prefix_of(path):
    """1001_123456.pdf -> '1001'"""
    return os.path.splitext(os.path.basename(path))[0].split("_")[0].strip()


def normalise_prefix(value):
    """Excel may store 1001 as 1001.0 -> '1001'."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s or None


def to_posix_rel(path):
    """Path as written in image_path: './01_real_data/<folder>/1001.png'"""
    rel = os.path.relpath(path).replace(os.sep, "/")
    return rel if rel.startswith(".") else f"./{rel}"


def parse_position(text):
    """'1: 30|385' -> (page, a, b) or None."""
    if pd.isna(text):
        return None
    m = POSITION_RE.match(str(text))
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def _inherited_box(doc, xref, key):
    """Read /CropBox or /MediaBox of a page, following /Parent inheritance."""
    seen = set()
    while xref and xref not in seen:
        seen.add(xref)
        typ, val = doc.xref_get_key(xref, key)
        if typ == "array":
            return [float(v) for v in val.strip("[]").split()]
        typ, val = doc.xref_get_key(xref, "Parent")
        xref = int(val.split()[0]) if typ == "xref" else None
    return None


def page_geometry(page):
    """Visible page box in PDF user space + rotation, needed to map PDF coords to pixels."""
    doc = page.parent
    box = _inherited_box(doc, page.xref, "CropBox") or _inherited_box(doc, page.xref, "MediaBox")
    if box is None:  # should not happen; fall back to PyMuPDF's view
        box = [0.0, 0.0, page.rect.width, page.rect.height]
    x0, y0, x1, y1 = box
    return dict(box_x0=min(x0, x1), box_y1=max(y0, y1), box_w=abs(x1 - x0), box_h=abs(y1 - y0),
                rotation=page.rotation % 360, page_w=page.rect.width, page_h=page.rect.height)


def render_first_page(pdf_path, png_path, dpi, render=True):
    """Render page 1 of the PDF to PNG (unless render=False). Returns image + page geometry."""
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        page = doc[0]
        geo = page_geometry(page)
        if render:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            pix.save(png_path)
            w, h = pix.width, pix.height
        else:
            with Image.open(png_path) as im:
                w, h = im.size
    return dict(png=png_path, w=w, h=h, **geo)


def to_image_xy(a, b, info, coord_unit, coord_order):
    """Convert one MAXQDA position (a|b) to (x, y) in PNG pixels."""
    if coord_order == "yx":
        a, b = b, a
    if coord_unit == "pixels":
        return a, b
    if coord_unit == "topleft_points":
        ux, uy = a, b
    else:  # maxqda_pdf: PDF user space, origin bottom-left
        ux = a - info["box_x0"]           # unrotated, origin top-left of visible box
        uy = info["box_y1"] - b
        W, H = info["box_w"], info["box_h"]
        ux, uy = {0: (ux, uy), 90: (H - uy, ux), 180: (W - ux, H - uy), 270: (uy, W - ux)}[info["rotation"]]
    return ux * info["w"] / info["page_w"], uy * info["h"] / info["page_h"]


def to_bbox(begin, end, info, coord_unit, coord_order):
    xa, ya = to_image_xy(begin[1], begin[2], info, coord_unit, coord_order)
    xb, yb = to_image_xy(end[1], end[2], info, coord_unit, coord_order)
    x1, x2 = sorted((xa, xb))
    y1, y2 = sorted((ya, yb))
    clip = lambda v, hi: int(round(max(0, min(v, hi))))
    return [clip(x1, info["w"]), clip(y1, info["h"]), clip(x2, info["w"]), clip(y2, info["h"])]


def draw_check_image(png_path, elements, out_path):
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 400)
    font_size = max(14, img.width // 80)
    try:
        font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    for el in elements:
        x1, y1, x2, y2 = el["bounding_box"]
        draw.rectangle([x1, y1, x2, y2], outline=(230, 30, 30), width=width)
        draw.text((x1 + width + 2, max(0, y1 - font_size - 4)), el["represents"], fill=(230, 30, 30), font=font)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=85)


def read_ground_truth(xlsx_path):
    df = pd.read_excel(xlsx_path, dtype=object)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in (COL_PREFIX, COL_CODE, COL_BEGIN, COL_END) if c not in df.columns]
    if missing:
        raise ValueError(f"{xlsx_path} is missing column(s): {missing}. Found: {list(df.columns)}")
    df["_prefix"] = df[COL_PREFIX].apply(normalise_prefix)
    df["_excel_row"] = df.index + 2  # header is row 1
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    logger = FolderLogger()
    root = args.root
    folders = sorted(
        d for d in glob(os.path.join(root, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith((".", "_"))
    )
    if not folders:
        print(f"No sub-folders found in {root}")
        return

    shared_gt_path = os.path.join(root, GT_FILE_NAME)
    shared_gt = read_ground_truth(shared_gt_path) if os.path.exists(shared_gt_path) else None

    for folder in folders:
        print(f"\n=== {folder} ===")

        # ---------- 1. PDFs -> PNGs ----------
        pdfs = {}  # prefix -> pdf path
        for pdf in sorted(glob(os.path.join(folder, "*.pdf")) + glob(os.path.join(folder, "*.PDF"))):
            p = prefix_of(pdf)
            if p in pdfs:
                logger.log(folder, "WARN", f"Duplicate prefix {p}: {os.path.basename(pdf)} ignored, "
                                           f"using {os.path.basename(pdfs[p])}")
                continue
            pdfs[p] = pdf

        images = {}  # prefix -> dict(png, w, h, page_w, page_h)
        for p, pdf in pdfs.items():
            png = os.path.join(folder, f"{p}.png")
            try:
                render = not (args.skip_existing and os.path.exists(png))
                images[p] = render_first_page(pdf, png, args.dpi, render=render)
            except Exception as e:
                logger.log(folder, "ERROR", f"Could not render {os.path.basename(pdf)}: {e}")
        print(f"Rendered {len(images)} PNG(s) from {len(pdfs)} PDF(s)")

        # ---------- 2. Ground truth -> JSON ----------
        local_gt_path = os.path.join(folder, GT_FILE_NAME)
        if os.path.exists(local_gt_path):
            gt_df, gt_source, gt_is_shared = read_ground_truth(local_gt_path), local_gt_path, False
        elif shared_gt is not None:
            gt_df, gt_source, gt_is_shared = shared_gt, shared_gt_path, True
        else:
            gt_df, gt_source, gt_is_shared = None, None, False
            logger.log(folder, "ERROR", f"No {GT_FILE_NAME} found in {folder} or {root}")

        gt_prefixes = set()
        if gt_df is not None:
            print(f"Ground truth from {gt_source}")
            for p, rows in gt_df[gt_df["_prefix"].notna()].groupby("_prefix", sort=True):
                if p not in images:
                    # A shared spreadsheet covers several folders: only report
                    # prefixes that have no PDF in ANY folder (done below).
                    if not gt_is_shared:
                        logger.log(folder, "MISSING_PDF",
                                   f"Ground truth for prefix {p} ({len(rows)} rows) but no PDF {p}_*.pdf")
                    gt_prefixes.add(p)
                    continue

                info = images[p]

                elements = []
                for _, r in rows.iterrows():
                    code = None if pd.isna(r[COL_CODE]) else str(r[COL_CODE]).strip()
                    b, e = parse_position(r[COL_BEGIN]), parse_position(r[COL_END])
                    where = f"prefix {p}, Excel row {r['_excel_row']}"
                    if not code:
                        logger.log(folder, "WARN", f"Empty Code ({where}) - row skipped")
                        continue
                    if b is None or e is None:
                        logger.log(folder, "WARN", f"Unreadable Beginning/End '{r[COL_BEGIN]}' / "
                                                   f"'{r[COL_END]}' ({where}) - row skipped")
                        continue
                    if b[0] != 1 or e[0] != 1:
                        logger.log(folder, "WARN", f"'{code}' is on page {b[0]}-{e[0]}, only page 1 is "
                                                   f"rendered ({where}) - row skipped")
                        continue
                    bbox = to_bbox(b, e, info, args.coord_unit, args.coord_order)
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        logger.log(folder, "WARN", f"'{code}' has an empty box after clipping {bbox} "
                                                   f"({where}) - row skipped")
                        continue
                    elements.append({
                        "represents": code,
                        "image_path": to_posix_rel(info["png"]),
                        "bounding_box": bbox,
                        "element_type": "Icon",
                    })

                if not elements:
                    logger.log(folder, "WARN", f"Prefix {p} has no usable ground-truth rows")
                    continue
                gt_prefixes.add(p)
                with open(os.path.join(folder, f"{p}_gt.json"), "w", encoding="utf-8") as f:
                    json.dump(elements, f, ensure_ascii=False)
                if args.check_images:
                    draw_check_image(info["png"], elements,
                                     os.path.join(CHECK_IMAGE_ROOT, os.path.basename(folder), f"{p}.jpg"))

        # ---------- 3. Log PDFs without ground truth ----------
        for p, pdf in pdfs.items():
            if p not in gt_prefixes:
                logger.log(folder, "MISSING_GT", f"{os.path.basename(pdf)} has no ground-truth rows "
                                                 f"for prefix {p}")

        # ---------- 4. Clean up ----------
        if not args.no_delete:
            for png in glob(os.path.join(folder, "*.png")):
                p = os.path.splitext(os.path.basename(png))[0]
                if p not in gt_prefixes or not os.path.exists(os.path.join(folder, f"{p}_gt.json")):
                    os.remove(png)
                    logger.log(folder, "DELETED", f"{os.path.basename(png)} (no ground truth)")
            for js in glob(os.path.join(folder, "*_gt.json")):
                p = os.path.basename(js)[: -len("_gt.json")]
                if p not in images:
                    os.remove(js)
                    logger.log(folder, "DELETED", f"{os.path.basename(js)} (no PDF)")

        n_pairs = len(glob(os.path.join(folder, "*_gt.json")))
        logger.log(folder, "INFO", f"{n_pairs} image/ground-truth pair(s) ready "
                                   f"(coord_unit={args.coord_unit}, coord_order={args.coord_order}, "
                                   f"dpi={args.dpi})")

    # Shared spreadsheet: prefixes with no PDF in any folder
    if shared_gt is not None:
        all_pdf_prefixes = {prefix_of(p) for f in folders for p in glob(os.path.join(f, "*.pdf"))}
        for p in sorted(set(shared_gt["_prefix"].dropna()) - all_pdf_prefixes):
            logger.log(root, "MISSING_PDF", f"Ground truth for prefix {p} in {shared_gt_path} "
                                            f"but no PDF {p}_*.pdf in any sub-folder")

    logger.write()
    print("\nDone. Logs written to", ", ".join(os.path.join(f, LOG_FILE_NAME) for f in logger.lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=REAL_DATA_ROOT)
    ap.add_argument("--dpi", type=int, default=RENDER_DPI)
    ap.add_argument("--coord-unit", choices=["maxqda_pdf", "topleft_points", "pixels"], default=COORD_UNIT)
    ap.add_argument("--coord-order", choices=["xy", "yx"], default=COORD_ORDER)
    ap.add_argument("--no-delete", action="store_true", help="only log, do not delete unmatched files")
    ap.add_argument("--skip-existing", action="store_true", help="do not re-render PNGs that already exist")
    ap.add_argument("--no-check-images", dest="check_images", action="store_false",
                    help=f"do not write box overlays to {CHECK_IMAGE_ROOT}/")
    main(ap.parse_args())
