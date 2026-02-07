import json

def open_reads_json_file(file_path : str) -> dict:
    try : 
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error: The file at {file_path} could not be read . {e}")
        return {}

def write_into_json (file_name : str , object_written : dict) -> dict :
    try :
        with open(file_name, 'w',encoding="latin-1") as f:
            json.dump(object_written, f, indent=2)
        with open(file_name, 'r',encoding="latin-1") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error: The file at {file_name} could not be written or read. {e}")
        return {}

def read_text_file(file_path : str) -> list[str] :
    try :
        with open(file_path, 'r', encoding="latin-1") as file:
            lines = file.readlines()
        return [ line.strip() for line in lines ]
    except Exception as e:
        print(f"Error: The file at {file_path} could not be read. {e}")
        return []
