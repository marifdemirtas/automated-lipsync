#!/bin/bash

# Set the directories containing the files
dir1="$1"
dir2="$2"

# Set the name of the Python script to call
python_script="api_v2.py"

# Iterate over each file in the first directory
for file1 in "$dir1"/*; do
  # Get the filename without the directory path
  filename=$(basename "$file1")

  # Concatenate the directory paths and filename to get the full path
  # of the file in the second directory
  corresponding_file=$(echo "$filename" | sed 's/.*_TO_//' | sed 's/....$//')
  file2="$dir2/$corresponding_file"

  # Call the Python script with the two files as arguments
  python "$python_script" "$file1" "$file2"
done

