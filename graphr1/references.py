"""Explicit reference downloads; other stages use only cached files."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def fetch_all(references, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, spec in references.items():
        path = directory / spec["filename"]
        if not path.exists():
            temporary = path.with_name(path.name + ".partial")
            try:
                with urlopen(spec["url"], timeout=120) as response, temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        manifest[name] = dict(url=spec["url"], filename=path.name, sha256=digest.hexdigest(), size=path.stat().st_size, recorded_at=datetime.now(timezone.utc).isoformat())
    (directory / "versions.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_hgnc(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", dtype=str)


def load_hallmark(path):
    result = {}
    with open(path) as source:
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError("Malformed GMT row")
            result[fields[0]] = fields[2:]
    return result


def load_string(path, chunksize=None):
    import pandas as pd
    return pd.read_csv(path, sep=r"\s+", chunksize=chunksize)


def load_string_info(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", dtype=str)
