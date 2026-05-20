import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path so we can import local modules
sys.path.append(os.getcwd())

from storage.gcs_storage import GCSStorage
from config.settings import GCS_KEY_PATH, GCS_BUCKET_NAME, GCS_PROJECT_ID

def test_gcs():
    print("=" * 50)
    print("GCS CONNECTION TESTER")
    print("=" * 50)

    # 1. Check .env
    load_dotenv()
    print(f"[*] Checking .env configuration:")
    print(f"    - Project ID : {GCS_PROJECT_ID}")
    print(f"    - Bucket Name: {GCS_BUCKET_NAME}")
    print(f"    - Key Path   : {GCS_KEY_PATH}")

    # 2. Check key file
    key_file = Path(GCS_KEY_PATH)
    if not key_file.exists():
        print(f"\n[!] ERROR: Key file not found at {GCS_KEY_PATH}")
        print(f"    Please place your JSON key in the 'secrets' folder and rename it to 'gcs_key.json'")
        return

    print(f"\n[*] Key file found. Attempting to connect...")

    # 3. Try to initialize storage
    try:
        gcs = GCSStorage()
        print("[+] GCS Client initialized successfully.")

        # 4. Try to list blobs
        print(f"[*] Fetching file list from bucket '{GCS_BUCKET_NAME}'...")
        blobs = list(gcs._bucket.list_blobs(max_results=10))
        
        if not blobs:
            print("    - Bucket is empty or no files found.")
        else:
            print(f"    - Found {len(blobs)} files (showing top 10):")
            for b in blobs:
                print(f"      - {b.name}")
        
        print("\n[SUCCESS] Your GCS connection is fully functional!")

    except Exception as e:
        print(f"\n[!] CONNECTION FAILED: {str(e)}")
        print("    Check your Project ID, Bucket Name, and Service Account permissions.")

if __name__ == "__main__":
    test_gcs()
