
# Load environment variables
source venv/bin/activate

# Start FastAPI server using uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000