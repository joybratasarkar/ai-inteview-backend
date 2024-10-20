from fastapi import WebSocket, WebSocketDisconnect
from pydub import AudioSegment, silence
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from base64 import b64encode
import logging

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=10)
from utils import (
    parse_resume, summarize_resume, generate_initial_question, AgentState, InterviewGraph,process_audio_blob,detect_silence
)
class InterviewBot:
    def __init__(self):
        self.interview_questions = []
        self.current_question_index = None
        self.candidate_responses = {}
        self.ongoing_transcription = ""
        self.chain_general = None
        self.temp_interview_questions = []
        self.accumulated_audio = AudioSegment.empty()
        self.accumulated_blobs = []

    # Ensure this method is not nested within __init__
    async def silence_detection(self, websocket: WebSocket):
        self.silence_detection_socket = websocket
        self.accumulated_blobs = []
        self.accumulated_audio = AudioSegment.empty()
    
        await websocket.accept()
    
        # Start a background task for sending periodic pings/heartbeats
        heartbeat_task = asyncio.create_task(self.send_heartbeat(websocket))
    
        try:
            while True:
                try:
                    audio_blob = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
                    loop = asyncio.get_running_loop()
                    audio_segment = await loop.run_in_executor(executor, process_audio_blob, audio_blob)
    
                    if audio_segment is None:
                        continue
    
                    self.accumulated_blobs.append(audio_blob)
                    self.accumulated_audio += audio_segment
    
                    silent_detected, cleaned_audio, overall_dBFS = await loop.run_in_executor(executor, detect_silence, self.accumulated_audio)
    
                    if silent_detected:
                        self.accumulated_audio = AudioSegment.empty()
                        self.accumulated_blobs = []
                        serialized_blobs = [b64encode(blob).decode('utf-8') for blob in self.accumulated_blobs]
                        response_data = {'silence_detected': silent_detected, 'completeBlob': serialized_blobs, 'overall_dBFS_int': overall_dBFS}
    
                        await websocket.send_text(json.dumps(response_data))
    
                    else:
                        await websocket.send_text(json.dumps({'silence_detected': False, 'completeBlob': [], 'overall_dBFS_int': overall_dBFS}))
    
                except asyncio.TimeoutError:
                    logger.info('No data received within timeout period, sending keep-alive message')
                    await websocket.send_text(json.dumps({'silence_detected': False, 'completeBlob': []}))
    
                except WebSocketDisconnect:
                    logger.info('Silence Detection WebSocket disconnected')
                    break
    
                except Exception as e:
                    logger.error(f"Error in silence_detection inner loop: {e}")
                    await websocket.send_text(json.dumps({'error': str(e)}))
    
        except WebSocketDisconnect:
            logger.info('Silence Detection WebSocket disconnected')
    
        except Exception as e:
            logger.error(f"Error in silence_detection outer loop: {e}")
    
        finally:
            heartbeat_task.cancel()  # Ensure heartbeat task is stopped when the WebSocket closes

    async def send_heartbeat(self, websocket: WebSocket, interval: float = 30.0):
        """Send a heartbeat message every `interval` seconds to keep the connection alive."""
        try:
            while True:
                # Simply send a heartbeat without checking specific state
                await websocket.send_text(json.dumps({"type": "heartbeat", "message": "keep-alive"}))
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            logging.info("Heartbeat task stopped due to WebSocket disconnect.")
        except Exception as e:
            logging.error(f"Error in send_heartbeat: {e}")

    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        try:
            state = AgentState()

            while True:
                data = await websocket.receive_text()
                logging.info(f"Received data: {data}")

                if data.startswith("UPLOAD_RESUME"):
                    pdf_data = await websocket.receive_bytes()
                    resume_text = parse_resume(pdf_data)
                    state.summary = await summarize_resume(resume_text)
                    logging.info(f"Resume summary: {state.summary}")
                    await websocket.send_text(json.dumps({"loader": False}))

                    graph = InterviewGraph(state)

                    start_message = graph.start_interview()
                    first_question = graph.ask_question()
                    state.last_question = first_question
                    await websocket.send_text(json.dumps({"question": first_question}))

                elif data.startswith("ANSWER"):
                    answer = data.split(":")[1].strip()
                    logging.info(f"Received answer: {answer}")
                    state.last_answer = answer

                    follow_up_question = graph.follow_up_question()
                    state.last_question = follow_up_question
                    await websocket.send_text(json.dumps({"question": follow_up_question}))

                elif data == "END":
                    end_message = graph.end_interview()
                    await websocket.send_text(json.dumps({"message": end_message}))
                    break

        except WebSocketDisconnect:
            logging.info("Client disconnected.")

# Create an instance of the InterviewBot class
interview_bot = InterviewBot()

silence_detection = interview_bot.silence_detection
websocket_endpoint = interview_bot.websocket_endpoint
