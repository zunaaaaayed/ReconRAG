"""Store uploaded PDFs safely before parsing them."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class StoredPdf:
    """Result of storing an uploaded PDF."""

    path: Path
    checksum: str
    created: bool


def store_pdf(
    content: bytes,
    original_filename: str,
    papers_dir: Path,
) -> StoredPdf:
    """Validate and store a PDF without overwriting a different paper."""
    filename = Path(original_filename).name

    if Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if b"%PDF-" not in content[:1024]:
        raise ValueError("The uploaded file does not appear to be a valid PDF.")

    checksum = sha256(content).hexdigest()

    papers_dir.mkdir(parents=True, exist_ok=True)
    destination = papers_dir / filename

    if destination.exists():
        existing_checksum = sha256(destination.read_bytes()).hexdigest()

        if existing_checksum == checksum:
            return StoredPdf(
                path=destination,
                checksum=checksum,
                created=False,
            )

        destination = papers_dir / (
            f"{Path(filename).stem}-{checksum[:8]}{Path(filename).suffix.lower()}"
        )

        if destination.exists():
            return StoredPdf(
                path=destination,
                checksum=checksum,
                created=False,
            )

    destination.write_bytes(content)

    return StoredPdf(
        path=destination,
        checksum=checksum,
        created=True,
    )
