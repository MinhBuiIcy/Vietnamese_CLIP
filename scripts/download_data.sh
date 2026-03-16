#!/usr/bin/env bash
# Download COCO 2017 images and Vietnamese captions.
#
# Images:   ~18 GB (train2017) + ~1 GB (val2017) from COCO servers
# Captions: loaded on-the-fly from HuggingFace datasets (no manual download needed)
#           Dataset: ai-enthusiasm-community/coco-2017-vietnamese

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_DIR/data"
IMAGES_DIR="$DATA_DIR/images"

mkdir -p "$IMAGES_DIR"

echo "=== Downloading COCO 2017 images ==="

# Train images (~18 GB)
if [ ! -d "$IMAGES_DIR/train2017" ]; then
  echo "Downloading train2017..."
  wget -q --show-progress -O /tmp/train2017.zip \
    http://images.cocodataset.org/zips/train2017.zip
  unzip -q /tmp/train2017.zip -d "$IMAGES_DIR"
  rm /tmp/train2017.zip
  echo "train2017 extracted."
else
  echo "train2017 already present, skipping."
fi

# Validation images (~1 GB)
if [ ! -d "$IMAGES_DIR/val2017" ]; then
  echo "Downloading val2017..."
  wget -q --show-progress -O /tmp/val2017.zip \
    http://images.cocodataset.org/zips/val2017.zip
  unzip -q /tmp/val2017.zip -d "$IMAGES_DIR"
  rm /tmp/val2017.zip
  echo "val2017 extracted."
else
  echo "val2017 already present, skipping."
fi

echo ""
echo "=== COCO 2017 images ready at $IMAGES_DIR ==="
echo ""
echo "NOTE: Vietnamese captions are streamed from HuggingFace at training time."
echo "      Dataset: ai-enthusiasm-community/coco-2017-vietnamese"
echo "      No manual download required for captions."
