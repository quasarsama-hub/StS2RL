"""Export source and measured evidence; external backends/game files stay local."""
from pathlib import Path
import argparse
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=root.parent / "spire2-rl-starter.zip")
    args = parser.parse_args()
    files = [root / name for name in ("README.md", "pyproject.toml", "backends.lock.json", ".gitignore")]
    for folder in ("src", "tests", "scripts", "docs"):
        files.extend(path for path in (root / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and not path.suffix == ".pyc"
                     and not any(part.endswith(".egg-info") for part in path.parts))
    for name in ("verification.json", "validation-report.json", "stall-trace.json"):
        artifact = root / "artifacts" / name
        if artifact.is_file():
            files.append(artifact)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(files)):
            archive.write(path, Path("spire2-rl") / path.relative_to(root))
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    print(f"{args.out}: {len(set(files))} files, {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
