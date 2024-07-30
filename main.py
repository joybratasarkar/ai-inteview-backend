import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from websocket_endpointfn import InterviewBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# app = FastAPI()
app = FastAPI(root_path="/ai-interviewer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an instance of the InterviewBot class
interview_bot = InterviewBot()

# WebSocket route for silence detection
@app.websocket("/silenceDetection")
async def silence_detection_handler(websocket: WebSocket):
    try:
        await interview_bot.silence_detection(websocket)
    except WebSocketDisconnect:
        logger.info("Silence Detection WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in silence_detection_handler: {e}")

# WebSocket route for the main interview process
@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    try:
        await interview_bot.websocket_endpoint(websocket)
    except WebSocketDisconnect:
        logger.info("Main WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket_handler: {e}")

# WebSocket route for question answering
@app.websocket("/questionAnswering")
async def question_answering_handler(websocket: WebSocket):
    try:
        await interview_bot.question_answering(websocket)
    except WebSocketDisconnect:
        logger.info("Question Answering WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in question_answering_handler: {e}")


@app.post("/close_sockets_main_socket")
async def close_connections():
    await interview_bot.close_sockets_main_socket()
    return {"message": "WebSocket connections closed"}

@app.post("/close_sockets_silence_detection_socket")
async def close_connections():
    await interview_bot.close_sockets_silence_detection_socket()
    return {"message": "WebSocket connections closed"}