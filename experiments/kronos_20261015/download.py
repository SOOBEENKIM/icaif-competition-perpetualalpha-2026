"""Fetch only pinned public upstream source and safetensors; no remote execution."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


def fetch(url, path=None):
    req = urllib.request.Request(url, headers={"User-Agent": "Kronos-reproduction-audit"})
    with urllib.request.urlopen(req, timeout=120) as response:
        if path is None:
            return response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".partial")
        with temp.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    dest = ROOT / "vendor" / "Kronos"
    if not dest.exists():
        data = fetch(f"https://codeload.github.com/shiyu-coder/Kronos/tar.gz/{COMMIT}")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                relative = Path(*Path(member.name).parts[1:])
                if not member.isfile():
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive path")
                path = dest / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    source_files = {str(p.relative_to(dest)): digest(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    source = dict(repository="https://github.com/shiyu-coder/Kronos", commit=COMMIT,
                  files=source_files, code_modified=False)
    (ROOT / "upstream.json").write_text(json.dumps(source, indent=2) + "\n")
    print("upstream", COMMIT, len(source_files), "files", flush=True)
    manifest = {}
    for name in ("Kronos-Tokenizer-base", "Kronos-base"):
        repo = "NeoQuasar/" + name
        meta = json.loads(fetch(f"https://huggingface.co/api/models/{repo}?blobs=true"))
        revision = meta["sha"]
        files = {}
        for entry in meta["siblings"]:
            filename = entry["rfilename"]
            if filename not in ("config.json", "model.safetensors", "README.md", "LICENSE"):
                continue
            path = ROOT / "models" / name / filename
            expected = entry.get("lfs", {}).get("sha256")
            if not path.exists():
                print("downloading", repo, revision, filename, flush=True)
                fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", path)
            sha = digest(path)
            if expected and sha != expected:
                raise ValueError("Published model digest mismatch: " + str(path))
            files[filename] = dict(sha256=sha, bytes=path.stat().st_size, published_sha256=expected)
        manifest[name] = dict(repo=repo, revision=revision, files=files,
                              created_at=meta.get("createdAt"), last_modified=meta.get("lastModified"))
        (ROOT / "model_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("verified", repo, revision, flush=True)
    paper = ROOT / "paper.html"
    if not paper.exists():
        fetch("https://arxiv.org/html/2508.02739v1", paper)


if __name__ == "__main__":
    main()
