"""Plant data: IBM AssetOpsBench sample data (Apache 2.0), pinned to one commit.

Eleven chillers with 14 years of work orders, events, alerts and anomalies, the failure-code
hierarchy and one month of Chiller 6 telemetry. Files are downloaded once and checked by size
and SHA-256, so every build of the plant database starts from the same bytes.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "IBM/AssetOpsBench"
COMMIT = "c6b96755b8a182aa090ee028852cbc83279b2982"  # branch main-0.x
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
LICENSE = "Apache-2.0"


@dataclass(frozen=True, slots=True)
class Source:
    path: str  # path inside the AssetOpsBench repository
    size: int
    sha256: str

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.path}"


_SAMPLE = "src/couchdb/sample_data"
SOURCES = (
    Source(
        f"{_SAMPLE}/work_order/all_wo_with_code_component_events.csv",
        939_899,
        "5d81ae9783c5012ea03f52046692f400c8e395007ca37f39c12115c861671789",
    ),
    Source(
        f"{_SAMPLE}/work_order/event.csv",
        695_569,
        "19313c21eb73eb942888842c9b0dfa23399a53be9128144ef6e74b4c8c210f61",
    ),
    Source(
        f"{_SAMPLE}/work_order/alert_events.csv",
        80_696,
        "ea16d2d7e5ab200c6d85d88bd5c609f4b804059e62bda2ae71f1ba90e18d07a2",
    ),
    Source(
        f"{_SAMPLE}/work_order/alert_rule.csv",
        972,
        "18f0780f56a6ece55a1c8f112d7c35c486d9cdbdb6598da51508295cd62e8977",
    ),
    Source(
        f"{_SAMPLE}/work_order/alert_rule_failure_code_mapping.csv",
        2_493,
        "6958e9ce72e511530334711d1a1a79dd78aa41951aa41d3d167c1ddd5a905265",
    ),
    Source(
        f"{_SAMPLE}/work_order/anomaly_to_failure_code_mapping.csv",
        1_806,
        "79e1ebab55e1506f9a69cc5d9c2d46af3928ca51a2f7d5e0c41286dbae512487",
    ),
    Source(
        f"{_SAMPLE}/work_order/component.csv",
        1_805,
        "77471c7310fbc8c823cd1dfa7b95b48cc004d59c9266def1d6a494de840c9f52",
    ),
    Source(
        f"{_SAMPLE}/work_order/failure_codes.csv",
        11_200,
        "d03e10bf74d93d7abb377a8a9458a89a5251ea40bb9b59ebc77900e39f130afb",
    ),
    Source(
        f"{_SAMPLE}/iot/chiller6_june2020_sensordata_couchdb.json",
        1_776_999,
        "8073af23ea71fc401aee97fd40b47678a80ad0549eb4153b1eb0ab8dc9b246e2",
    ),
    Source(
        "src/servers/fmsr/failure_modes.yaml",
        564,
        "0eb75cc501e6d2f5cde5561c7ef12bdfa9d75260bf0e57148b04fc99c90fdc44",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(source: Source, path: Path) -> bool:
    return path.exists() and path.stat().st_size == source.size and _sha256(path) == source.sha256


def download(source: Source, raw_dir: Path) -> Path:
    """Fetches one file unless a verified copy is already present."""
    dest = raw_dir / source.filename
    if verify(source, dest):
        return dest
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(source.url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    if not verify(source, partial):
        partial.unlink(missing_ok=True)
        raise OSError(f"{source.filename}: size or checksum mismatch (expected {source.sha256})")
    partial.replace(dest)
    return dest


def fetch_all(raw_dir: Path) -> dict[str, Path]:
    return {s.filename: download(s, raw_dir) for s in SOURCES}
