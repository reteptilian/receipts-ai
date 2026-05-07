from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

from receipts_ai import download_google_drive_folder


class FakeDriveRequest:
    def __init__(self, response: Any):
        self.response = response

    def execute(self) -> Any:
        return self.response


class FakeDriveFilesResource:
    def __init__(self, service: FakeDriveService):
        self.service = service

    def list(
        self,
        *,
        q: str,
        fields: str,
        pageSize: int,
        pageToken: str | None,
        supportsAllDrives: bool,
        includeItemsFromAllDrives: bool,
    ) -> FakeDriveRequest:
        del fields, pageSize, supportsAllDrives, includeItemsFromAllDrives
        folder_id = q.split("'")[1]
        self.service.list_calls.append((folder_id, pageToken))
        pages = self.service.folder_pages[folder_id]
        page_index = int(pageToken or "0")
        response: dict[str, Any] = {"files": pages[page_index]}
        next_page_index = page_index + 1
        if next_page_index < len(pages):
            response["nextPageToken"] = str(next_page_index)
        return FakeDriveRequest(response)

    def get_media(self, *, fileId: str, supportsAllDrives: bool) -> FakeDriveRequest:
        del supportsAllDrives
        self.service.download_calls.append(fileId)
        return FakeDriveRequest(self.service.file_content[fileId])

    def export_media(self, *, fileId: str, mimeType: str) -> FakeDriveRequest:
        self.service.export_calls.append((fileId, mimeType))
        return FakeDriveRequest(self.service.file_content[fileId])


class FakeDriveService:
    def __init__(
        self,
        *,
        folder_pages: dict[str, list[list[dict[str, str]]]],
        file_content: dict[str, bytes],
    ):
        self.folder_pages = folder_pages
        self.file_content = file_content
        self.list_calls: list[tuple[str, str | None]] = []
        self.download_calls: list[str] = []
        self.export_calls: list[tuple[str, str]] = []

    def files(self) -> FakeDriveFilesResource:
        return FakeDriveFilesResource(self)


def test_google_drive_folder_id_accepts_raw_id():
    assert download_google_drive_folder.google_drive_folder_id("folder-123") == "folder-123"


@pytest.mark.parametrize(
    ("url", "folder_id"),
    [
        ("https://drive.google.com/drive/folders/folder-123?usp=sharing", "folder-123"),
        ("https://drive.google.com/open?id=folder-456", "folder-456"),
    ],
)
def test_google_drive_folder_id_extracts_url_folder_id(url: str, folder_id: str):
    assert download_google_drive_folder.google_drive_folder_id(url) == folder_id


def test_download_google_drive_folder_downloads_paginated_folder_files(tmp_path: Path):
    service = FakeDriveService(
        folder_pages={
            "source-folder": [
                [
                    {
                        "id": "file-1",
                        "name": "receipt.pdf",
                        "mimeType": "application/pdf",
                        "md5Checksum": "8b04d5e3775d298e78455efc5ca404d5",
                    }
                ],
                [
                    {
                        "id": "file-2",
                        "name": "receipt.pdf",
                        "mimeType": "application/pdf",
                        "md5Checksum": "a9f0e61a137d86aa9db53465e0801612",
                    }
                ],
            ]
        },
        file_content={"file-1": b"first", "file-2": b"second"},
    )

    downloaded = download_google_drive_folder.download_google_drive_folder(
        source_folder="source-folder",
        destination=tmp_path,
        drive_service=service,
    )

    assert service.list_calls == [("source-folder", None), ("source-folder", "1")]
    assert service.download_calls == ["file-1", "file-2"]
    assert [file.output_path.name for file in downloaded] == ["receipt.pdf", "receipt (2).pdf"]
    assert (tmp_path / "receipt.pdf").read_bytes() == b"first"
    assert (tmp_path / "receipt (2).pdf").read_bytes() == b"second"


def test_download_google_drive_folder_skips_regular_file_with_existing_hash(tmp_path: Path):
    existing_path = tmp_path / "already-here.pdf"
    existing_path.write_bytes(b"same content")
    service = FakeDriveService(
        folder_pages={
            "source-folder": [
                [
                    {
                        "id": "file-1",
                        "name": "receipt.pdf",
                        "mimeType": "application/pdf",
                        "md5Checksum": _md5_hex(b"same content"),
                    }
                ]
            ]
        },
        file_content={"file-1": b"same content"},
    )

    downloaded = download_google_drive_folder.download_google_drive_folder(
        source_folder="source-folder",
        destination=tmp_path,
        skip_existing_by_hash=True,
        drive_service=service,
    )

    assert downloaded == []
    assert service.download_calls == []
    assert not (tmp_path / "receipt.pdf").exists()


def test_download_google_drive_folder_uses_downloaded_hash_for_later_source_files(
    tmp_path: Path,
):
    service = FakeDriveService(
        folder_pages={
            "source-folder": [
                [
                    {
                        "id": "file-1",
                        "name": "receipt.pdf",
                        "mimeType": "application/pdf",
                        "md5Checksum": _md5_hex(b"same content"),
                    },
                    {
                        "id": "file-2",
                        "name": "renamed.pdf",
                        "mimeType": "application/pdf",
                        "md5Checksum": _md5_hex(b"same content"),
                    },
                ]
            ]
        },
        file_content={"file-1": b"same content", "file-2": b"same content"},
    )

    downloaded = download_google_drive_folder.download_google_drive_folder(
        source_folder="source-folder",
        destination=tmp_path,
        skip_existing_by_hash=True,
        drive_service=service,
    )

    assert service.download_calls == ["file-1"]
    assert [file.output_path.name for file in downloaded] == ["receipt.pdf"]
    assert (tmp_path / "receipt.pdf").read_bytes() == b"same content"
    assert not (tmp_path / "renamed.pdf").exists()


def test_download_google_drive_folder_exports_google_workspace_files(tmp_path: Path):
    service = FakeDriveService(
        folder_pages={
            "source-folder": [
                [
                    {
                        "id": "sheet-1",
                        "name": "Budget",
                        "mimeType": "application/vnd.google-apps.spreadsheet",
                        "md5Checksum": _md5_hex(b"xlsx bytes"),
                    }
                ]
            ]
        },
        file_content={"sheet-1": b"xlsx bytes"},
    )

    downloaded = download_google_drive_folder.download_google_drive_folder(
        source_folder="source-folder",
        destination=tmp_path,
        drive_service=service,
    )

    assert service.export_calls == [
        (
            "sheet-1",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    ]
    assert downloaded[0].output_path == tmp_path / "Budget.xlsx"
    assert (tmp_path / "Budget.xlsx").read_bytes() == b"xlsx bytes"


def test_download_google_drive_folder_recurses_into_nested_folders(tmp_path: Path):
    service = FakeDriveService(
        folder_pages={
            "source-folder": [
                [
                    {
                        "id": "nested-folder",
                        "name": "Receipts/2026",
                        "mimeType": download_google_drive_folder.GOOGLE_DRIVE_FOLDER_MIME_TYPE,
                    }
                ]
            ],
            "nested-folder": [
                [
                    {
                        "id": "file-1",
                        "name": "receipt.pdf",
                        "mimeType": "application/pdf",
                    }
                ]
            ],
        },
        file_content={"file-1": b"receipt"},
    )

    downloaded = download_google_drive_folder.download_google_drive_folder(
        source_folder="source-folder",
        destination=tmp_path,
        recursive=True,
        drive_service=service,
    )

    assert downloaded[0].output_path == tmp_path / "Receipts_2026" / "receipt.pdf"
    assert (tmp_path / "Receipts_2026" / "receipt.pdf").read_bytes() == b"receipt"


def test_download_google_drive_folder_main_passes_oauth_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[dict[str, object]] = []

    def fake_download_google_drive_folder(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-download-google-drive-folder",
            "source-folder",
            str(tmp_path),
            "--recursive",
            "--skip-existing-by-hash",
            "--google-oauth-credentials",
            "/tmp/credentials.json",
            "--google-oauth-authorized-user",
            "/tmp/authorized_user.json",
        ],
    )
    monkeypatch.setattr(
        download_google_drive_folder,
        "download_google_drive_folder",
        fake_download_google_drive_folder,
    )

    download_google_drive_folder.main()

    assert calls == [
        {
            "source_folder": "source-folder",
            "destination": tmp_path,
            "recursive": True,
            "skip_existing_by_hash": True,
            "oauth_credentials_path": Path("/tmp/credentials.json"),
            "oauth_authorized_user_path": Path("/tmp/authorized_user.json"),
        }
    ]


def _md5_hex(content: bytes) -> str:
    return hashlib.md5(content, usedforsecurity=False).hexdigest()
