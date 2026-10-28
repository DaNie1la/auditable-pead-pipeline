from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT / "deliverables" / "github" / "pead-portfolio"
ARCHIVE_PATH = PROJECT_ROOT / "deliverables" / "github" / "pead-portfolio-github.zip"

ROOT_FILES = [
    ".gitignore",
    "README.md",
    "PORTFOLIO.md",
    "ATTRIBUTION.md",
    "REPRODUCIBILITY.md",
    "GITHUB_PUBLISHING.md",
    "WORKSPACE_MAP.md",
    "pytest.ini",
    "requirements.txt",
    "requirements-test.txt",
    "requirements-phase3.txt",
    "requirements-phase4.txt",
    "requirements-phase5.txt",
]

REPORT_FILES = [
    "phase1_data_quality.md",
    "phase2_transcript_quality.md",
    "phase3_finbert_quality.md",
    "phase4_market_quality.md",
    "phase4_fundamental_quality.md",
    "phase5_modeling_quality.md",
    "phase5_results_interpretation.md",
]

FORBIDDEN_SUFFIXES = {
    ".docx",
    ".pdf",
    ".pkl",
    ".zip",
    ".gz",
    ".csv",
    ".safetensors",
}
FORBIDDEN_TEXT = [str(Path.home()), str(Path.home()).replace("\\", "/")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(relative: Path) -> None:
    source = PROJECT_ROOT / relative
    if not source.exists():
        raise FileNotFoundError(source)
    destination = RELEASE_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if RELEASE_ROOT.exists() or ARCHIVE_PATH.exists():
        raise FileExistsError(
            "release output already exists; move or remove the exact release artifacts before rebuilding"
        )
    RELEASE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FILES:
        copy_file(Path(name))
    for path in sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml")):
        copy_file(path.relative_to(PROJECT_ROOT))
    for path in sorted((PROJECT_ROOT / "src" / "pead_pipeline").glob("*.py")):
        copy_file(path.relative_to(PROJECT_ROOT))
    for path in sorted((PROJECT_ROOT / "scripts").glob("run_phase*.py")):
        copy_file(path.relative_to(PROJECT_ROOT))
    copy_file(Path("scripts") / "build_portfolio_release.py")
    for path in sorted((PROJECT_ROOT / "tests").glob("test_phase*.py")):
        copy_file(path.relative_to(PROJECT_ROOT))
    for name in REPORT_FILES:
        copy_file(Path("reports") / name)

    release_files = sorted(path for path in RELEASE_ROOT.rglob("*") if path.is_file())
    for path in release_files:
        if path.suffix.lower() in {".py", ".md", ".txt", ""}:
            content = path.read_text(encoding="utf-8", errors="replace")
            sanitized = content
            for local_root in {
                str(PROJECT_ROOT),
                str(PROJECT_ROOT).replace("\\", "/"),
                str(Path.home()),
                str(Path.home()).replace("\\", "/"),
            }:
                sanitized = sanitized.replace(local_root, "<LOCAL_PATH>")
            if sanitized != content:
                path.write_text(sanitized, encoding="utf-8")

    violations: list[str] = []
    for path in release_files:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden suffix: {path.relative_to(RELEASE_ROOT)}")
        if path.suffix.lower() in {".py", ".md", ".txt", ""}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in FORBIDDEN_TEXT:
                if needle in text:
                    violations.append(
                        f"local personal path in {path.relative_to(RELEASE_ROOT)}"
                    )
    if violations:
        raise RuntimeError("release privacy audit failed: " + "; ".join(violations))

    manifest_lines = [
        f"{sha256_file(path)}  {path.relative_to(RELEASE_ROOT).as_posix()}"
        for path in release_files
    ]
    (RELEASE_ROOT / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    shutil.make_archive(str(ARCHIVE_PATH.with_suffix("")), "zip", RELEASE_ROOT)
    print(
        f"portfolio_release_complete files={len(release_files) + 1} "
        f"archive={ARCHIVE_PATH}"
    )


if __name__ == "__main__":
    main()
