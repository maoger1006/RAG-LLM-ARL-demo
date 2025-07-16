import json
import os
import numpy as np


def count_unique_video_names(json_file_path: str) -> int:
    """
    Reads a JSON file containing a list of dictionaries, counts the number
    of unique values associated with the 'video_name' key.

    Args:
        json_file_path: The path to the JSON file.

    Returns:
        The count of unique video names, or 0 if an error occurs.
    """
    unique_names = set()

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            # Load the entire JSON structure (expected to be a list)
            data = json.load(f)
            print(len(data))
            # Ensure the loaded data is a list
            if not isinstance(data, list):
                print(f"Error: JSON file '{json_file_path}' does not contain a list.")
                return 0

            # Iterate through each dictionary in the list
            for item in data:
                # Check if the item is a dictionary and has the 'video_name' key
                if isinstance(item, dict):
                    video_name = item.get('video_name') # Use .get() for safety
                    if video_name is not None:
                        unique_names.add(video_name)
                else:
                    # Optional: Print a warning if an item is not a dictionary
                    print(f"Warning: Skipping non-dictionary item in JSON list: {item}")

        # The number of unique names is the size of the set
        return len(unique_names)

    except FileNotFoundError:
        print(f"Error: File not found at '{json_file_path}'")
        return 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Check file format.")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 0
    
count_unique_video_names("/home/mingyang/video_benchmark/MMBench-Video/MMBench-Video_q.json")