from fastapi import WebSocket, WebSocketDisconnect
from pydub import AudioSegment, silence
import numpy as np
from utils import process_audio_blob, detect_silence, is_general_question, create_prompt, detect_silence, extract_project_details, get_project_info
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompt import chain_interview, general_prompt_template, llm, output_parser, chain_interview_end
import json
import warnings

from base64 import b64encode
import torchaudio
import io

from io import BytesIO
import whisper

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class InterviewBot:
    def __init__(self):
        self.interview_questions = []
        self.current_question_index = None
        self.candidate_responses = {}
        self.ongoing_transcription = ""
        self.chain_general = None
        self.temp_interview_questions = [ "Can you explain the difference between a module and a namespace in TypeScript?",
    "How would you implement an authentication system in a Ruby on Rails application?"]

    async def silence_detection(self, websocket: WebSocket):
        await websocket.accept()
        accumulated_audio = AudioSegment.silent(duration=0, frame_rate=16000)
        accumulated_blobs = []
    
        try:
            while True:
                try:
                    audio_blob = await websocket.receive_bytes()
                    with open("received_audio.wav", "wb") as f:
                        f.write(audio_blob)
                    audio_segment = process_audio_blob(audio_blob)
                    if audio_segment is None:
                        continue
                    accumulated_blobs.append(audio_blob)
                    accumulated_audio += audio_segment
                    silent_detected = detect_silence(accumulated_audio)
                    if silent_detected:
                        accumulated_audio = AudioSegment.empty()
                        serialized_blobs = [b64encode(blob).decode('utf-8') for blob in accumulated_blobs]
                        response_data = {'silence_detected': silent_detected, 'completeBlob': serialized_blobs}
                        await websocket.send_text(json.dumps(response_data))
                        accumulated_blobs = []
                except WebSocketDisconnect:
                    print('Silence Detection Socket disconnected')
                    break
                except Exception as e:
                    print(f"Error: {e}")
    
        except WebSocketDisconnect:
            print('Silence Detection Socket disconnected')
        except Exception as e:
            print(f"Error: {e}")
            
        


    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                if data:
                    job_id = data.get('job_id')
                    projectId = data.get('projectId')
                    project_details = await get_project_info()
                    details = extract_project_details(project_details)
                    prompt_update = create_prompt(details, general_prompt_template)
                    general_prompt = ChatPromptTemplate.from_template(prompt_update)
                    self.chain_general = general_prompt | llm | output_parser

                    if self.interview_questions:
                        first_question = self.interview_questions[self.current_question_index]
                        await websocket.send_text(f"{first_question}")
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            await websocket.send_text(f"Error: {e}")

    async def question_answering(self, websocket: WebSocket):
        await websocket.accept()
        self.current_question_index = 1

        try:
            while True:
                data = await websocket.receive_json()
                if data:
                    job_id = data.get('projectId')
                    sender = data.get('sender')
                    text = data.get('text')

                    if self.current_question_index < len(self.temp_interview_questions):
                        question = self.temp_interview_questions[self.current_question_index]
                        self.candidate_responses[question] = text
                        scores = is_general_question(text)
                        general_score = scores.get('general', 0)
                        question_score = scores.get('question', 0)
                        answer_score = scores.get('answer', 0)
                        Unanswered = scores.get('Unanswered', 0)

                        response = None
                        if answer_score > max(general_score, question_score, Unanswered):
                            response = chain_interview.invoke({'response': text})
                        elif question_score > max(general_score, answer_score, Unanswered):
                            response = chain_interview.invoke({'response': text})
                        elif Unanswered > max(general_score, question_score, answer_score):
                            response = chain_interview.invoke({'response': text})
                        elif general_score > max(Unanswered, question_score, answer_score):
                            response = self.chain_general.invoke({'context': text})

                        if response:
                            cleaned_string = question.replace("-", "")
                            merged_message = f"{response}{cleaned_string}"
                            self.current_question_index += 1
                            if general_score > max(question_score, answer_score, Unanswered):
                                await websocket.send_text(f"{response}")
                            else:
                                await websocket.send_text(f"Next Question {merged_message}")
                        else:
                            combined_responses = " ".join(self.candidate_responses.values())
                            response = chain_interview_end.invoke({'summary': combined_responses})
                            await websocket.send_json(response)
                    elif general_score > max(Unanswered, question_score, answer_score):
                        response = chain_interview.invoke({'response': text})
                        await websocket.send_text(f"{response}")
                    else:
                        combined_responses = " ".join(self.candidate_responses.values())
                        response = chain_interview_end.invoke({'summary': combined_responses})
                        await websocket.send_json(response)
                        break

        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            await websocket.send_text(f"Error: {e}")

# Create an instance of the InterviewBot class
interview_bot = InterviewBot()

silence_detection = interview_bot.silence_detection
websocket_endpoint = interview_bot.websocket_endpoint
question_answering = interview_bot.question_answering
