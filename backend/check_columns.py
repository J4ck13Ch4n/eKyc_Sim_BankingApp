import requests
import sys

URL = "https://ekyc-backend-436637848640.asia-northeast1.run.app/api/admin/check_schema"

try:
    resp = requests.get(URL)
    data = resp.json()
    cols = data.get("columns", [])
    print(f"Columns found: {cols}")
    
    missing = []
    if "email" not in cols: missing.append("email")
    if "password_hash" not in cols: missing.append("password_hash")
    
    if missing:
        print(f"❌ MISSING COLUMNS: {missing}")
        exit(1)
    else:
        print("✅ All columns present")
        exit(0)
except Exception as e:
    print(f"Error: {e}")
    exit(1)
