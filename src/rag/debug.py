import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def dump_retrieval(query: str, retrieved: list[dict], out_dir: str = "logs/retrieval") -> str:
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


def dump_response(
    query: str,
    response_text: str,
    citations: list[str | None] | None = None,
    out_dir: str = "logs/responses",
) -> str:
    """
    Persist the final LLM response for later inspection.

    Args:
        query: The original user query.
        response_text: The text returned by the LLM.
        citations: Optional list of cited sources or filenames.
        out_dir: Directory (relative to project root) to write logs.

    Returns:
        Absolute path to the written response file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"{time_string}.json"

    payload = {
        "query": query,
        "response": response_text,
        "citations": citations or [],
        "timestamp": time_string,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return str(path)

