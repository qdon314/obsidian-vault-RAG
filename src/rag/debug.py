import json
from pathlib import Path
from datetime import datetime

def dump_retrieval(query: str, retrieved: list[dict], out_dir: str = "./logs/retrieval") -> str:
    """
    Dumps the retrieval results to a JSON file in the specified output directory.

    Args:
        query (str): The query string.
        retrieved (list[dict]): A list of dictionaries representing the retrieved items.
        out_dir (str, optional): The output directory where the JSON file will be saved. Defaults to "./logs/retrieval".
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"{time_string}.json"
    payload = {
        "query": query,
        "retrieved": retrieved,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
