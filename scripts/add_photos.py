#!/usr/bin/env python3
"""
Usage: python scripts/add_photos.py <directory> <category>

Scans <directory> for image files, generates WebP thumbnails, updates
_data/photography.yml, and prints the R2 upload commands.

Requires: pip install Pillow

Example:
  python scripts/add_photos.py ~/Desktop/new-shots norway
"""

import sys
import os
import re
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("Error: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

YAML_PATH = Path(__file__).parent.parent / '_data' / 'photography.yml'
THUMB_DIR = Path(__file__).parent.parent / '_temp'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG', '.JPEG', '.PNG', '.GIF'}
THUMB_MAX_WIDTH = 900
THUMB_QUALITY = 82


def load_yaml_raw(path):
    with open(path, 'r') as f:
        return f.read()


def save_yaml_raw(path, content):
    with open(path, 'w') as f:
        f.write(content)


def get_base_url(content):
    m = re.search(r'^base_url:\s*"([^"]+)"', content, re.MULTILINE)
    return m.group(1) if m else ''


def get_folder_for_category(content, category):
    pattern = rf'  {re.escape(category)}:.*?folder:\s*"([^"]*)"'
    m = re.search(pattern, content, re.DOTALL)
    return m.group(1) if m else category


def get_existing_photos(content, category):
    pattern = rf'  {re.escape(category)}:.*?photos:(.*?)(?=\n  \w|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return set()
    return set(re.findall(r'- (.+)', match.group(1)))


def append_photos(content, category, new_photos):
    pattern = rf'(  {re.escape(category)}:.*?photos:(?:.*?\n)*?)((?=\n  \w)|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Error: category '{category}' not found in {YAML_PATH}")
        print("Available categories:")
        for m in re.finditer(r'^  (\w+):', content, re.MULTILINE):
            if m.group(1) not in ('base_url',):
                print(f"  {m.group(1)}")
        sys.exit(1)
    insert_pos = match.end(1)
    lines = '\n'.join(f'      - {p}' for p in new_photos)
    return content[:insert_pos] + lines + '\n' + content[insert_pos:]


def make_thumbnail(src_path, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
        w, h = img.size
        if w > THUMB_MAX_WIDTH:
            img = img.resize((THUMB_MAX_WIDTH, int(h * THUMB_MAX_WIDTH / w)), Image.LANCZOS)
        img.save(dest_path, 'WEBP', quality=THUMB_QUALITY)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    directory, category = sys.argv[1], sys.argv[2]
    directory = Path(os.path.expanduser(directory))

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    images = sorted(f.name for f in directory.iterdir() if f.suffix in IMAGE_EXTS)

    if not images:
        print(f"No image files found in {directory}")
        sys.exit(0)

    content = load_yaml_raw(YAML_PATH)
    existing = get_existing_photos(content, category)
    folder = get_folder_for_category(content, category)
    base_url = get_base_url(content)

    new_photos = [img for img in images if img not in existing]

    if not new_photos:
        print(f"All {len(images)} photos already in '{category}' — nothing to add.")
        sys.exit(0)

    # Generate thumbnails
    print(f"Generating {len(new_photos)} WebP thumbnail(s)...")
    thumb_folder = THUMB_DIR / folder
    for photo in new_photos:
        src = directory / photo
        thumb_name = Path(photo).stem + '.webp'
        dest = thumb_folder / thumb_name
        make_thumbnail(src, dest)
        print(f"  {photo} → thumbnails/{folder}/{thumb_name}")

    # Update YAML
    content = append_photos(content, category, new_photos)
    save_yaml_raw(YAML_PATH, content)
    print(f"\nAdded {len(new_photos)} photo(s) to '{category}' in photography.yml")

    # Print R2 upload instructions
    r2_prefix = base_url.split('.r2.dev/', 1)[-1] if '.r2.dev/' in base_url else ''
    print(f"\nUpload originals to R2 ({r2_prefix}/{folder}/):")
    for photo in new_photos:
        print(f"  {directory / photo}")

    print(f"\nUpload thumbnails to R2 ({r2_prefix}/thumbnails/{folder}/):")
    for photo in new_photos:
        thumb_name = Path(photo).stem + '.webp'
        print(f"  {thumb_folder / thumb_name}")


if __name__ == '__main__':
    main()
