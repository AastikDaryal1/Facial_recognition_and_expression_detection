"""
storage/gcs_storage.py
───────────────────────
Thin wrapper around google-cloud-storage.

Security note
─────────────
The original notebook embedded the service-account private key directly
in source code — a serious credential leak risk.  This module reads the
key file PATH from the environment variable GCS_KEY_PATH instead.
Never hard-code credentials in source files.
"""

import os
import zipfile
from pathlib import Path
from typing import List, Optional

from google.cloud import storage
from google.oauth2 import service_account

from config.settings import GCS_BUCKET_NAME, GCS_KEY_PATH, GCS_PROJECT_ID
from utils.logger import get_logger

log = get_logger(__name__)


class GCSStorage:
    """Manages all interactions with the Google Cloud Storage bucket."""

    def __init__(
        self,
        key_path: str = GCS_KEY_PATH,
        project_id: str = GCS_PROJECT_ID,
        bucket_name: str = GCS_BUCKET_NAME,
    ) -> None:
        self._bucket_name = bucket_name
        self._client = self._build_client(key_path, project_id)
        self._bucket = self._client.bucket(bucket_name)
        log.info("GCSStorage initialised", extra={"bucket": bucket_name})

    def delete_person_data(self, person: any, TEAM_FACES_DIR: Path) -> None:
        """Delete local directory and GCS folder for a person."""
        # Delete local directory containing the person's photos
        safe_name = person.full_name.replace("/", "_").replace("..", "_")
        local_dir = TEAM_FACES_DIR / safe_name
        if local_dir.is_dir():
            import shutil
            try:
                shutil.rmtree(local_dir, ignore_errors=True)
                log.info("Deleted local person directory %s", local_dir)
            except Exception as e:
                log.warning("Failed to delete local directory %s: %s", local_dir, e)

        # Delete GCS folder for the person
        try:
            gcs_prefix = f"team_faces/{safe_name}/"
            self.delete_folder(gcs_prefix)
        except Exception as e:
            log.warning("GCS cleanup failed for %s: %s", person.id, e)

    # ── Construction ───────────────────────────────────────────────────────
    @staticmethod
    def _build_client(key_path: str, project_id: str) -> storage.Client:
        """
        Build a GCS client.
        Priority:
          1. Service-account JSON file (GCS_KEY_PATH)
          2. Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS
             or gcloud auth — works on GCP VMs / Cloud Run automatically)
        """
        key_file = Path(key_path)
        if key_file.exists():
            creds = service_account.Credentials.from_service_account_file(str(key_file))
            log.info("Using service-account key: %s", key_file)
            return storage.Client(credentials=creds, project=project_id)
        else:
            log.info("Key file not found — falling back to Application Default Credentials")
            return storage.Client(project=project_id)

    # ── Existence check ───────────────────────────────────────────────────
    def blob_exists(self, remote_path: str) -> bool:
        return self._bucket.blob(remote_path).exists()

    def get_blob_updated(self, remote_path: str):
        """Return the 'updated' timestamp of a blob, or None if it doesn't exist."""
        blob = self._bucket.get_blob(remote_path)
        return blob.updated if blob else None

    def get_latest_blob_updated(self, prefix: str):
        """Find the newest 'updated' timestamp among all blobs with this prefix."""
        blobs = list(self._bucket.list_blobs(prefix=prefix))
        if not blobs:
            return None
        return max(b.updated for b in blobs)

    def all_blobs_exist(self, remote_paths: List[str]) -> bool:
        return all(self.blob_exists(p) for p in remote_paths)

    # ── Download ──────────────────────────────────────────────────────────
    def download_file(self, remote_path: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading gs://%s/%s → %s", self._bucket_name, remote_path, local_path)
        self._bucket.blob(remote_path).download_to_filename(str(local_path))
        log.info("Downloaded %s (%.1f MB)", local_path.name, local_path.stat().st_size / 1e6)
        return local_path

    def download_folder(self, remote_prefix: str, local_dir: str | Path) -> None:
        """Recursively download all blobs under a prefix."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        blobs = self._bucket.list_blobs(prefix=remote_prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            
            # e.g. team_faces/Ojas/img.jpg -> local_dir/Ojas/img.jpg
            relative_path = blob.name[len(remote_prefix):].lstrip("/")
            dest_path = local_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            log.debug("Downloading %s → %s", blob.name, dest_path)
            blob.download_to_filename(str(dest_path))
        log.info("Folder download complete: %s", remote_prefix)

    def download_and_extract_zip(
        self,
        remote_path: str,
        local_zip: str | Path,
        extract_to: str | Path,
    ) -> Path:
        """Download a zip from GCS and extract it locally."""
        local_zip  = self.download_file(remote_path, local_zip)
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s → %s", local_zip, extract_to)
        with zipfile.ZipFile(str(local_zip), "r") as zf:
            zf.extractall(str(extract_to))
        log.info("Extraction complete: %s", extract_to)
        return extract_to

    # ── Upload ────────────────────────────────────────────────────────────
    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        local_path = Path(local_path)
        log.info("Uploading %s → gs://%s/%s", local_path, self._bucket_name, remote_path)
        self._bucket.blob(remote_path).upload_from_filename(str(local_path))
        log.info("Upload complete: %s", remote_path)

    def delete_folder(self, prefix: str) -> None:
        """Delete all blobs under the given prefix (folder) in the bucket."""
        blobs = list(self._bucket.list_blobs(prefix=prefix))
        if not blobs:
            log.debug("No blobs found for prefix %s to delete.", prefix)
            return
        for blob in blobs:
            try:
                blob.delete()
                log.info("Deleted GCS blob %s", blob.name)
            except Exception as e:
                log.warning("Failed to delete blob %s: %s", blob.name, e)
        log.info("Completed deletion of GCS folder %s (deleted %d blobs)", prefix, len(blobs))

