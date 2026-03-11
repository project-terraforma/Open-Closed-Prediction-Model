@echo off
start "Backend" cmd /k "cd stillopen\backend && venv\Scripts\activate && uvicorn app.main:app --reload --port 8000"
start "Frontend" cmd /k "cd stillopen\frontend && npm run dev"
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
