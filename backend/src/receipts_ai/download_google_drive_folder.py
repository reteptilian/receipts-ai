from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

LOGGER = logging.getLogger(__name__)

GOOGLE_DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
GOOGLE_WORKSPACE_EXPORTS: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.drawing": ("image/png", ".png"),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}


@dataclass(frozen=True)
class DownloadedDriveFile:
    drive_id: str
    drive_name: str
    output_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download files from a Google Drive folder using OAuth authentication."
    )
    parser.add_argument(
        "source_folder",
        help="Google Drive source folder ID or URL.",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Local destination folder to write downloaded files into.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Download files from nested folders too.",
    )
    parser.add_argument(
        "--google-oauth-credentials",
        type=Path,
        help="Path to OAuth client credentials JSON. Defaults to gspread's config path.",
    )
    parser.add_argument(
        "--google-oauth-authorized-user",
        type=Path,
        help="Path to OAuth authorized user JSON. Defaults to gspread's config path.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        help="Show logs at this level.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s")

    download_google_drive_folder(
        source_folder=args.source_folder,
        destination=args.destination,
        recursive=args.recursive,
        oauth_credentials_path=args.google_oauth_credentials,
        oauth_authorized_user_path=args.google_oauth_authorized_user,
    )


def download_google_drive_folder(
    *,
    source_folder: str,
    destination: Path,
    recursive: bool = False,
    oauth_credentials_path: Path | None = None,
    oauth_authorized_user_path: Path | None = None,
    drive_service: Any | None = None,
) -> list[DownloadedDriveFile]:
    folder_id = google_drive_folder_id(source_folder)
    destination.mkdir(parents=True, exist_ok=True)
    service = drive_service or _create_google_drive_service(
        oauth_credentials_path=oauth_credentials_path,
        oauth_authorized_user_path=oauth_authorized_user_path,
    )
    downloaded = _download_drive_folder_files(
        service,
        folder_id=folder_id,
        destination=destination,
        recursive=recursive,
    )
    LOGGER.info(
        "Downloaded %d Google Drive file(s) from folder %s to %s",
        len(downloaded),
        folder_id,
        destination,
    )
    return downloaded


def google_drive_folder_id(source_folder: str) -> str:
    parsed = urlparse(source_folder)
    if parsed.scheme and parsed.netloc:
        query_id = parse_qs(parsed.query).get("id")
        if query_id and query_id[0]:
            return query_id[0]

        match = re.search(r"/folders/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)

        raise ValueError(f"Could not find a Google Drive folder ID in URL: {source_folder}")

    return source_folder


def _create_google_drive_service(
    *,
    oauth_credentials_path: Path | None,
    oauth_authorized_user_path: Path | None,
) -> Any:
    from googleapiclient.discovery import build  # pyright: ignore[reportUnknownVariableType]

    credentials = _create_google_oauth_credentials(
        oauth_credentials_path=oauth_credentials_path,
        oauth_authorized_user_path=oauth_authorized_user_path,
    )
    return cast(Any, build("drive", "v3", credentials=credentials))


def _create_google_oauth_credentials(
    *,
    oauth_credentials_path: Path | None,
    oauth_authorized_user_path: Path | None,
) -> Any:
    import gspread

    kwargs: dict[str, Any] = {}
    if oauth_credentials_path is not None:
        kwargs["credentials_filename"] = oauth_credentials_path
    if oauth_authorized_user_path is not None:
        kwargs["authorized_user_filename"] = oauth_authorized_user_path
    client = gspread.oauth(**kwargs)
    return client.http_client.auth


def _download_drive_folder_files(
    service: Any,
    *,
    folder_id: str,
    destination: Path,
    recursive: bool,
) -> list[DownloadedDriveFile]:
    downloaded: list[DownloadedDriveFile] = []
    used_paths: set[Path] = set()
    _download_drive_folder_files_into(
        service,
        folder_id=folder_id,
        destination=destination,
        recursive=recursive,
        used_paths=used_paths,
        downloaded=downloaded,
    )
    return downloaded


def _download_drive_folder_files_into(
    service: Any,
    *,
    folder_id: str,
    destination: Path,
    recursive: bool,
    used_paths: set[Path],
    downloaded: list[DownloadedDriveFile],
) -> None:
    for drive_file in _list_drive_folder_children(service, folder_id=folder_id):
        drive_id = drive_file["id"]
        drive_name = drive_file["name"]
        mime_type = drive_file.get("mimeType")
        if mime_type == GOOGLE_DRIVE_FOLDER_MIME_TYPE:
            if recursive:
                subfolder_destination = _unique_output_path(
                    destination / _safe_filename(drive_name),
                    used_paths=used_paths,
                )
                subfolder_destination.mkdir(parents=True, exist_ok=True)
                _download_drive_folder_files_into(
                    service,
                    folder_id=drive_id,
                    destination=subfolder_destination,
                    recursive=recursive,
                    used_paths=used_paths,
                    downloaded=downloaded,
                )
            else:
                LOGGER.info("Skipping nested Google Drive folder %s", drive_name)
            continue

        output_path = _download_drive_file(
            service,
            drive_id=drive_id,
            drive_name=drive_name,
            mime_type=mime_type,
            destination=destination,
            used_paths=used_paths,
        )
        downloaded.append(
            DownloadedDriveFile(
                drive_id=drive_id,
                drive_name=drive_name,
                output_path=output_path,
            )
        )


def _list_drive_folder_children(service: Any, *, folder_id: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if page_token is None:
            return files


def _download_drive_file(
    service: Any,
    *,
    drive_id: str,
    drive_name: str,
    mime_type: str | None,
    destination: Path,
    used_paths: set[Path],
) -> Path:
    export = GOOGLE_WORKSPACE_EXPORTS.get(mime_type or "")
    if export is None:
        request = service.files().get_media(fileId=drive_id, supportsAllDrives=True)
        output_path = _unique_output_path(
            destination / _safe_filename(drive_name),
            used_paths=used_paths,
        )
    else:
        export_mime_type, extension = export
        request = service.files().export_media(fileId=drive_id, mimeType=export_mime_type)
        output_path = _unique_output_path(
            destination / _safe_filename_with_extension(drive_name, extension),
            used_paths=used_paths,
        )

    content = request.execute()
    output_path.write_bytes(content)
    LOGGER.info("Downloaded Google Drive file %s to %s", drive_name, output_path)
    return output_path


def _safe_filename(filename: str) -> str:
    safe = filename.replace("/", "_").replace("\\", "_").strip()
    return safe or "untitled"


def _safe_filename_with_extension(filename: str, extension: str) -> str:
    safe = _safe_filename(filename)
    if Path(safe).suffix:
        return safe
    return f"{safe}{extension}"


def _unique_output_path(path: Path, *, used_paths: set[Path]) -> Path:
    if path not in used_paths and not path.exists():
        used_paths.add(path)
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem} ({index}){suffix}")
        if candidate not in used_paths and not candidate.exists():
            used_paths.add(candidate)
            return candidate

    raise RuntimeError(f"Could not find an unused output path for {path}")


if __name__ == "__main__":
    main()
