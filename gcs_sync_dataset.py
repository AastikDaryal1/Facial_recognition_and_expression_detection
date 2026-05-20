import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

from storage.gcs_storage import GCSStorage
from config.settings import TEAM_FACES_DIR, GCS_BUCKET_NAME

def sync_dataset():
    print("=" * 60)
    print("GCS DATASET SYNC TOOL")
    print("=" * 60)

    load_dotenv()
    TEAM_FACES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gcs = GCSStorage()
        print(f"[*] Connected to bucket: {GCS_BUCKET_NAME}")
        
        # We only want files under 'team_faces/'
        prefix = "team_faces/"
        print(f"[*] Searching for images with prefix '{prefix}'...")
        
        blobs = list(gcs._bucket.list_blobs(prefix=prefix))
        print(f"[*] Found {len(blobs)} potential files.")

        download_count = 0
        skipped_count = 0

        for blob in blobs:
            if blob.name.endswith('/'): # Skip folders
                continue
            
            # Map GCS path to local path
            # GCS: team_faces/MemberName/image.jpg -> Local: data/raw/TeamFaces/MemberName/image.jpg
            relative_path = blob.name.replace(prefix, "", 1)
            if not relative_path: continue
            
            local_path = TEAM_FACES_DIR / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if not local_path.exists():
                print(f"    [SYNC] Downloading: {blob.name} ...", end="\r")
                blob.download_to_filename(str(local_path))
                download_count += 1
            else:
                skipped_count += 1

        print(f"\n\n[SUCCESS] Sync complete!")
        print(f"    - Downloaded: {download_count} files")
        print(f"    - Skipped   : {skipped_count} files (already present)")
        print(f"    - Local Dir : {TEAM_FACES_DIR}")

    except Exception as e:
        print(f"\n[!] SYNC FAILED: {str(e)}")

if __name__ == "__main__":
    sync_dataset()
