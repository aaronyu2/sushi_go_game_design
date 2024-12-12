#!/bin/bash

# Set the target directory
TARGET_DIR="collected_data"

# Check if the directory exists
if [ -d "$TARGET_DIR" ]; then
    # Find all directories inside the target directory and remove them
    find "$TARGET_DIR" -mindepth 1 -type d -exec rm -rf {} +

    echo "All folders inside '$TARGET_DIR' have been removed."
else
    echo "'$TARGET_DIR' directory does not exist."
fi

# reset file numbering
TARGET_FILE="collected_data/filename"
echo -n 1 > "$TARGET_FILE"
