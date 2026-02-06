import json

def open_json_file(file_path : str) -> dict:
    try : 
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}

