import json

def open_reads_json_file(file_path : str) -> dict:
    try : 
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}

def write_into_json (file_name : str , object_written : dict) -> None :
    with open(file_name, 'w',encoding="utf-8") as f:
        json.dump(object_written, f, indent=2)
    return
