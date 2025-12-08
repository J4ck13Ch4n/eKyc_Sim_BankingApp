# Backend Guide: Local Run & Cloud Testing

This document explains how to run the backend locally for development and how to test the currently deployed instance on Google Cloud Run.

## 1. Local Development (SQLite)

Run the server on your machine for debugging or development. The local version uses a file-based `ekyc.db` (SQLite).

### Prerequisites
- Python 3.9+
- Virtualenv (recommended)

### Quick Start
1.  **Install dependencies:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2.  **Run Server:**
    ```bash
    uvicorn main:app --reload --port 8001
    ```

3.  **Access:**
    - API: `http://localhost:8001`
    - Docs: `http://localhost:8001/docs`

---

## 2. Testing Deployed Backend

The backend is currently LIVE at:
> **URL:** `https://ekyc-backend-436637848640.asia-northeast1.run.app`
> **Database:** Supabase (PostgreSQL)

### Automated Test Script
We have provided a script `test_deployment.py` to verify the end-to-end flow (ZKP Generation -> Enrollment -> DB Persistence).

1.  **Run the test:**
    ```bash
    python test_deployment.py
    ```

2.  **Check Results:**
    - The script prints the status to the terminal and saves detailed logs to `test_log.txt`.
    - **Success** means:
        1. A ZKP proof was generated locally.
        2. It was successfully sent to the Cloud Run server.
        3. The server verified it and saved the user to Supabase.
        4. The script queried the server back and found the user.

---

## 3. Deployment Reference

Use this command ONLY if you have modified the code and want to update the live version.

```powershell
gcloud run deploy ekyc-backend --source . --region asia-northeast1
```

*Note: The Database connection (`DATABASE_URL`) is already configured securely on Cloud Run, so you don't need to provide it again.*
