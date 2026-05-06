from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from receipts_ai.config import config_value, first_config_value

if TYPE_CHECKING:
    from firebase_admin import App

DEFAULT_FIRESTORE_COLLECTION = "transactions"
DEFAULT_FIREBASE_EMULATOR_PROJECT_ID = "receipts-ai-local"
FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR = "FIREBASE_SERVICE_ACCT_KEY_FILEPATH"
FIRESTORE_EMULATOR_HOST_ENV_VAR = "FIRESTORE_EMULATOR_HOST"
FIRESTORE_PROJECT_ID_ENV_VARS = (
    "FIREBASE_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
)

LOGGER = logging.getLogger(__name__)


class FirestoreDocumentReference(Protocol):
    def set(self, document_data: dict[str, Any], *, merge: bool = False) -> object: ...


class FirestoreCollectionReference(Protocol):
    def document(self, document_id: str) -> FirestoreDocumentReference: ...


class FirestoreClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionReference: ...


def create_firestore_client() -> FirestoreClient:
    emulator_host = config_value(FIRESTORE_EMULATOR_HOST_ENV_VAR)
    service_account_key_filepath = config_value(FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR)

    if emulator_host:
        os.environ[FIRESTORE_EMULATOR_HOST_ENV_VAR] = emulator_host
        LOGGER.info(
            "Creating Firestore client for emulator at %s using project %s",
            emulator_host,
            _firestore_project_id(),
        )
        return _create_firestore_emulator_client()

    if service_account_key_filepath:
        LOGGER.info(
            "Creating Firestore client from service account file %s",
            service_account_key_filepath,
        )
        return _create_firestore_service_account_client(Path(service_account_key_filepath))

    raise RuntimeError(
        "Set FIRESTORE_EMULATOR_HOST to use the local Firestore emulator or "
        "FIREBASE_SERVICE_ACCT_KEY_FILEPATH to use production Cloud Firestore."
    )


def _create_firestore_emulator_client() -> FirestoreClient:
    from firebase_admin import firestore
    from google.auth.credentials import AnonymousCredentials

    project_id = _firestore_project_id()
    cred = AnonymousCredentials()
    LOGGER.debug("Initializing Firebase app for Firestore emulator with project %s", project_id)
    app = _firebase_app(
        name="receipts-ai-firestore-emulator", options={"projectId": project_id}, credential=cred
    )
    return cast(FirestoreClient, cast(object, firestore.client(app=app)))


def _create_firestore_service_account_client(service_account_key_filepath: Path) -> FirestoreClient:
    from firebase_admin import credentials, firestore

    if not service_account_key_filepath.is_file():
        raise RuntimeError(
            f"{FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR} must point to a service account JSON file"
        )

    LOGGER.debug("Initializing Firebase app with service account credentials")
    credential = credentials.Certificate(str(service_account_key_filepath))
    app = _firebase_app(name="receipts-ai-firestore-service-account", credential=credential)
    return cast(FirestoreClient, cast(object, firestore.client(app=app)))


def _firebase_app(
    *,
    name: str,
    credential: Any | None = None,
    options: dict[str, str] | None = None,
) -> App:
    import firebase_admin

    try:
        return cast("App", firebase_admin.get_app(name))
    except ValueError:
        initialize_app = cast(  # type: ignore[reportUnknownMemberType]
            "Callable[[Any | None, dict[str, str] | None, str], App]",
            firebase_admin.initialize_app,
        )
        return initialize_app(credential, options, name)


def _firestore_project_id() -> str:
    return (
        first_config_value(FIRESTORE_PROJECT_ID_ENV_VARS, DEFAULT_FIREBASE_EMULATOR_PROJECT_ID)
        or DEFAULT_FIREBASE_EMULATOR_PROJECT_ID
    )
