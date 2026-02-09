import fitz
from pathlib import Path
from tokenizer.normalizer import normalizer_
import json

def load_data_set (folder_path : str) -> list[str] :
    files_set : list[str] = []
    try :
        folder = Path(folder_path)
        for file in folder.glob("*") :
            files_set.append(file.name)
        return files_set
    except Exception as e :
        print(f"Error Occured !!{e}")


def extract_text_from_files (file_path : str) -> None :
    try :
        with open("data_set/data.json", 'r', encoding="utf-8") as f:
            data = json.load(f)
        doc = fitz.open(file_path)
        counter : int = 0
        for page in doc :
            normalized_text = normalizer_(page.get_text())
            # print(f"normalized_text : {normalized_text} ")
            for normalized_word in normalized_text :
                data["data_set"].append(normalized_word)
            normalized_text.clear()
            counter += 1

        with open("data_set/data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    except Exception as e :
        print(f"Error Occured !!{e}")
        return None
