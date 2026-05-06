# Face & Emotion Detection System - Local Setup Guide

This guide provides step-by-step instructions for setting up and running the Face & Emotion Detection system on a local machine.

## 1. Prerequisites

Before starting, ensure you have the following installed:
- **Python 3.10.x**: [Download Python 3.10](https://www.python.org/downloads/windows/) (Add to PATH during installation).
- **Node.js (v18 or later)**: [Download Node.js](https://nodejs.org/).
- **Git**: [Download Git](https://git-scm.com/).

---

## 2. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/AastikDaryal1/Facial_recognition_and_expression_detection.git
cd Facial_recognition_and_expression_detection
```

### Step 2: Backend Setup (Python)
Create and activate a virtual environment, then install dependencies:
```powershell
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install specific bcrypt version for authentication compatibility
pip install bcrypt==4.0.1
```

### Step 3: Frontend Setup (Node.js)
```bash
cd frontend
npm install
cd ..
```

---

## 3. Configuration (.env files)

You must create two `.env` files for the system to function correctly. **Do not share these files on GitHub.**

### Backend `.env` (Create in the Root Directory)
Create a file named `.env` and paste the following:
```env
API_KEY=netsmartz.net
ENV=development
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_DIR=logs
```

### Frontend `.env` (Create in the `frontend/` Directory)
Create a file named `.env` and paste the following:
```env
VITE_API_KEY=netsmartz.net
VITE_API_BASE_URL=http://localhost:8000
VITE_MAX_UPLOAD_SIZE_MB=5
```

---

## 4. Running the Application

You will need two terminal windows/tabs open.

### Terminal 1: Start the Backend
```powershell
.\venv\Scripts\python main.py api
```
*The API will be available at http://localhost:8000*

### Terminal 2: Start the Frontend
```bash
cd frontend
npm run dev
```
*The Dashboard will be available at http://localhost:5173*

---

## 5. Usage & Authentication

- **Initial Login**: When you open the frontend, click **"Log In"**.
- **Password**: Enter `netsmartz.net` to gain access to the dashboard.
- **Admin Access**: If the system asks for a username/password for admin functions, use:
    - **Username**: `admin`
    - **Password**: `netsmartz.net`

## 6. Troubleshooting
- **File Access Errors**: If you encounter errors related to log files, ensure no other instance of the app is running. Use `taskkill /F /IM python.exe /T` to clear lingering processes.
- **Database**: The system uses a local SQLite database (`face_app.db`). It is created automatically on the first run.
