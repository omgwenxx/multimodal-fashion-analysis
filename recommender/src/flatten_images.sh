#!/bin/bash

# Define source and destination directories
SOURCE_DIR="images"
DEST_DIR="images_flat"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all image files in the source directory and copy them to the destination
find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" \) -exec cp {} "$DEST_DIR" \;

echo "All images have been flattened into the '$DEST_DIR' directory."
