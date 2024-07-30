from fastapi import WebSocket, WebSocketDisconnect
from pydub import AudioSegment, silence
from utils import process_audio_blob, is_general_question, format_project_details, create_prompt,detect_silence,process_batch, extract_project_details,detect_silence_ffmpeg, questionAnswerArray, get_Interview_Question, get_project_info,summarize_text
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompt import chain_interview, general_prompt_template, llm, output_parser, chain_interview_end
import json
import warnings
import logging
from base64 import b64encode
import torchaudio
import io
from io import BytesIO
import whisper
from transformers import pipeline
import time
import asyncio

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class InterviewBot:
    def __init__(self):
        self.interview_questions = []
        self.current_question_index = None
        self.candidate_responses = {}
        self.ongoing_transcription = ""
        self.chain_general = None
        self.temp_interview_questions = []
        self.silence_detection_socket = None
        self.main_socket = None
        self.accumulated_blobs = []
        self.accumulated_audio = AudioSegment.silent(duration=0, frame_rate=16000)
        self.start_time = time.time()
    async def silence_detection(self, websocket: WebSocket):
        self.silence_detection_socket = websocket
        await websocket.accept()
        try:
            while True:
                try:
                    # Use asyncio.wait_for to set a timeout for receiving data
                    audio_blob = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
                    # Process audio blob in-memory to avoid unnecessary file I/O
                    audio_segment = process_audio_blob(audio_blob)
                    if audio_segment is None:
                        continue
                    
                    self.accumulated_blobs.append(audio_blob)
                    self.accumulated_audio += audio_segment
                    print('self.accumulated_audio',len(self.accumulated_audio))
                    silent_detected = detect_silence(self.accumulated_audio)
                    print('silent_detected',silent_detected)
                    if silent_detected:
                        print('silent_detected',silent_detected)
                        self.accumulated_audio = AudioSegment.empty()
                        serialized_blobs = [b64encode(blob).decode('utf-8') for blob in self.accumulated_blobs]
                        response_data = {'silence_detected': silent_detected, 'completeBlob': serialized_blobs}
                        await websocket.send_text(json.dumps(response_data))
                        self.accumulated_blobs = []
                    else:
                        # Send keep-alive message
                        await websocket.send_text(json.dumps({'silence_detected': False, 'completeBlob': []}))
                except asyncio.TimeoutError:
                    logger.info('No data received within timeout period, sending keep-alive message')
                    await websocket.send_text(json.dumps({'silence_detected': False, 'completeBlob': []}))
                except WebSocketDisconnect:
                    logger.info('Silence Detection Socket disconnected')
                    break
                except Exception as e:
                    logger.error(f"Error in silence_detection inner loop: {e}")
                    await websocket.send_text(json.dumps({'error': str(e)}))
        except WebSocketDisconnect:
            logger.info('Silence Detection Socket disconnected')
        except Exception as e:
            logger.error(f"Error in silence_detection outer loop: {e}")



    async def websocket_endpoint(self, websocket: WebSocket):
        self.main_socket = websocket
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                if data:
                    job_id = data.get('job_id')
                    projectId = data.get('projectId')
                    project_info = await get_Interview_Question(projectId)
                    self.interview_questions = await questionAnswerArray(project_info)
                    logger.info(f'project_info: {self.interview_questions}')
                    self.temp_interview_questions = self.interview_questions
                    project_details = await get_project_info(projectId)
                    details = extract_project_details(project_details)
                    logger.info(f'details: {details}')
                    prompt_update = create_prompt(details, general_prompt_template)
                    general_prompt = ChatPromptTemplate.from_template(prompt_update)
                    self.chain_general = general_prompt | llm | output_parser

                    if self.interview_questions:
                        first_question = self.interview_questions[0]
                        await websocket.send_text(f"{first_question}")
        except WebSocketDisconnect:
            logger.info("Main WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in websocket_endpoint: {e}")
            await websocket.send_text(f"Error: {e}")

    async def close_sockets(self):
        if self.silence_detection_socket:
            await self.silence_detection_socket.close()
            logger.info("Closed Silence Detection WebSocket")

        if self.main_socket:
            await self.main_socket.close()
            logger.info("Closed Main WebSocket")

    async def question_answering(self, websocket: WebSocket):
        await websocket.accept()
        self.current_question_index = 1
        labels = [
            'interview_question', 
            'user_answer', 
            'no_answer', 
            'repeat_request', 
            'move_to_the_next_question'
        ]

        # Dictionary to cache frequently asked questions and their responses
        cache = {}

        try:
            while True:
                data = await websocket.receive_json()
                if not data:
                    continue
                
                job_id = data.get('projectId')
                sender = data.get('sender')
                text = data.get('text')

                if self.current_question_index < len(self.temp_interview_questions):
                    question = self.temp_interview_questions[self.current_question_index]
                    self.candidate_responses[question] = text

                    start_time = time.time()  # Start timing

                    # Process batch of texts
                    texts = [text]  # Modify this list as needed
                    summarized_texts = await process_batch(texts)
                    summarized_text = summarized_texts[0]  # Assuming single text input

                    scores = is_general_question(summarized_text)  # Function to determine scores
                    scores_dict = {label: scores.get(label, 0) for label in labels}
                    max_score_label = max(scores_dict, key=scores_dict.get)

                    response = None
                    if max_score_label == 'user_answer':
                        response = await asyncio.to_thread(chain_interview.invoke, {'response': summarized_text})
                    elif max_score_label == 'repeat_request':
                        question_to_send = self.temp_interview_questions[self.current_question_index - 1] if self.current_question_index > 0 else question
                        await websocket.send_text(f"Ok, I will repeat the question: {question_to_send}")
                    elif max_score_label == 'interview_question':
                        response = await asyncio.to_thread(chain_interview.invoke, {'response': question})
                    elif max_score_label == 'move_to_the_next_question':
                        response = await asyncio.to_thread(chain_interview.invoke, {'response': summarized_text})
                        self.current_question_index += 1
                    elif max_score_label in ['follow_up', 'no_answer']:
                        response = await asyncio.to_thread(self.chain_general.invoke, {'context': summarized_text})

                    end_time = time.time()  # End timing
                    elapsed_time = end_time - start_time  # Calculate elapsed time
                    print(f"Time taken from summarize to generate response: {elapsed_time:.2f} seconds")

                    if response:
                        message = f"{response} {question.replace('-', '')}" if max_score_label != 'follow_up' else response
                        await websocket.send_text(message)
                        self.current_question_index += 1

                else:
                    combined_responses = " ".join(self.candidate_responses.values())
                    texts = [combined_responses]  # Batch processing combined responses
                    summarized_responses = await process_batch(texts)
                    summarized_response = summarized_responses[0]  # Assuming single combined response

                    response = await asyncio.to_thread(chain_interview_end.invoke, {'summary': summarized_response})
                    await websocket.send_json(response)

        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            await websocket.send_text(f"Error: {e}")


    

# Create an instance of the InterviewBot class
interview_bot = InterviewBot()

silence_detection = interview_bot.silence_detection
websocket_endpoint = interview_bot.websocket_endpoint
question_answering = interview_bot.question_answering
