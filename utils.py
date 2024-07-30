from pydub import AudioSegment, silence
import numpy as np
from transformers import pipeline
import httpx
import librosa
import scipy.signal as signal
import torch
import asyncio
import configparser
import logging
from pydub.utils import make_chunks
from pydub import AudioSegment
import subprocess
import os


current_question_index = 0
temp_interview_questions:[]
# output_parser = StrOutputParser()
# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')


# Set the current environment
current_env = "dev"  # Change this to "staging" or "live" as needed
logger = logging.getLogger(__name__)

# Retrieve URLs for the current environment
job_management_service_v2_url = config[current_env]['job_management_service_v2']
job_management_service_v1_url = config[current_env]['job_management_service_v1']
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

classifier = pipeline("zero-shot-classification", model="facebook/bart-base")
# wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
# tokenizer = Wav2Vec2Tokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

def process_audio_blob(audio_blob):
    try:
        # Convert raw audio bytes to numpy array (assuming it's PCM encoded)
        audio_array = np.frombuffer(audio_blob, dtype=np.int16)
        
        # Create AudioSegment from numpy array (assuming it's mono, 16-bit PCM)
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=16000,  # Adjust based on your audio settings
            sample_width=2,    # 16-bit audio
            channels=1         # Mono audio
        )
        
        return audio_segment
    except Exception as e:
        logger.error(f"Error processing audio blob: {e}")
        return None

def detect_silence_ffmpeg(audio_segment, min_silence_len=5000, silence_thresh=-50):
    try:
        # Export audio segment to a temporary file
        temp_audio_path = "temp_audio.wav"
        audio_segment.export(temp_audio_path, format="wav")
        
        # Construct the FFmpeg command
        command = [
            "ffmpeg", "-v", "verbose", "-i", temp_audio_path, "-af",
            f"silencedetect=n={silence_thresh}dB:d={min_silence_len / 1000}", "-f", "null", "-"
        ]
        
        # Run the command and capture the output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Decode the output
        s = stderr.decode("utf-8")
        lines = s.split('\n')
        
        # Extract silence start and end points
        start, end = [], []
        for line in lines:
            if 'silence_start' in line:
                start.append(float(line.split(':')[1].strip()))
            elif 'silence_end' in line:
                end.append(float(line.split('|')[0].split(':')[1].strip()))
        
        silent_ranges = list(zip(start, end))
        logger.debug('silent_ranges: %s', silent_ranges)
        
        # Clean up temporary file
        os.remove(temp_audio_path)
        
        # Check if any silence segment is longer than the specified threshold
        for s, e in silent_ranges:
            if (e - s) >= min_silence_len / 1000:
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error detecting silence: {e}")
        return False

    
# def detect_silence(audio_segment, min_silence_len=3000, silence_thresh=-30):
#     try:
#         # Perform silence detection on the audio segment
#         silent_ranges = silence.detect_silence(
#             audio_segment, 
#             min_silence_len=min_silence_len, 
#             silence_thresh=silence_thresh,
            
#         )
#         # print('silent_ranges',silent_ranges)
#         # Check if any silence segment is longer than 4 seconds (4000 ms)
#         for start, end in silent_ranges:
#             if (end - start) >= min_silence_len:
#                 return True
        
#         return False
#     except Exception as e:
#         logger.error(f"Error detecting silence: {e}")
#         return False
def detect_silence(audio_segment, min_silence_len=3000, silence_thresh=-30):
    try:
        chunk_size = 10000  # 10 seconds chunks
        silent_detected = False
        
        for start in range(0, len(audio_segment), chunk_size):
            end = min(start + chunk_size, len(audio_segment))
            chunk = audio_segment[start:end]
            silent_ranges = silence.detect_silence(
                chunk,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            for start, end in silent_ranges:
                if (end - start) >= min_silence_len:
                    silent_detected = True
                    break
            
            if silent_detected:
                break
            
        return silent_detected
    except Exception as e:
        logger.error(f"Error detecting silence: {e}")
        return False

    

    
def is_general_question(question):
    # candidate_labels = ['general', 'question', 'answer','no-answer','dont-know-the-answer','move-to-next-question']
    # candidate_labels = ['general', 'question', 'answer','Unanswered','rephrase']
    candidate_labels = [
        'interview_question', 
        'user_answer', 
        'no_answer', 
        'repeat_request', 
        'move_to_the_next_question'
    ]
            # 'general_question', 

            # 'follow_up',
        # 'off_topic', 
        # 'clarification_request', 

    
    result = classifier(question, candidate_labels=candidate_labels, multi_label=True,device=0)
    print('Classifier result:', result) 
    # Get the scores for the labels
    scores = {label: score for label, score in zip(result['labels'], result['scores'])} 

    return  scores
    






def create_prompt(project_details, context):
    formatted_project_details = format_project_details(project_details)
    context_with_project_details = f"Project Information:\n{formatted_project_details}"
    
    new_prompt = f"""
    Answer the following general question.
    You are an AI bot of Supersourcing from Indore, India.
    You can answer a wide range of questions, but you prefer to keep the conversation related to interviews.
    kept it brief to point .
    Give the answer in 20 words max in brief.
    {context_with_project_details}
    <context>
    {context}
    </context>
    """

    return new_prompt.strip()



def format_project_details(project_details):
    client_name = project_details.get('client_name', 'N/A')
    role = project_details.get('role', 'N/A')
    experience_range = project_details.get('experience_range', 'N/A')
    max_experience = project_details.get('max_experience', 'N/A')
    min_experience = project_details.get('min_experience', 'N/A')
    hiring_type = project_details.get('hiring_type', 'N/A')
    tentative_start = project_details.get('tentative_start', 'N/A')
    secondary_skills = project_details.get('secondary_skills', [])
    primary_skills = project_details.get('primary_skills', [])
    working_time_zone = project_details.get('working_time_zone', 'N/A')
    tools_used = project_details.get('tools_used', [])
    no_of_rounds = project_details.get('no_of_rounds', 'N/A')
    interview_rounds = ", ".join([f"Round {round.get('round_number', 'N/A')}: {round.get('round_name', 'N/A')}" for round in project_details.get('interview_rounds', [])])
    communication_skill = project_details.get('communication_skill', 'N/A')

    formatted_details = (
        f"The project is for a client named {client_name}. "
        f"The role is {role} with an experience range of {experience_range} years. "
        f"The maximum experience required is {max_experience} years and the minimum is {min_experience} years. "
        f"The hiring type is {hiring_type} and the tentative start date is {tentative_start}. "
        f"The secondary skills required are {', '.join(secondary_skills)}, and the primary skills required are {', '.join(primary_skills)}. "
        f"The working time zone is {working_time_zone}. The tools used include {', '.join(tools_used)}. "
        f"There are {no_of_rounds} rounds of interviews: {interview_rounds}. "
        f"The required communication skill level is {communication_skill}."
    )
    return formatted_details



def extract_project_details(response):
    if not response['success']:
        return None

    data = response['data'][0]  

    project_details = {
        'client_name': data.get('client_name', ''),
        'role': data.get('role', [{}])[0].get('role', ''),
        'experience_range': data.get('experience_range', [{}])[0].get('data', ''),
        'max_experience': data.get('max_experience', ''),
        'min_experience': data.get('min_experience', ''),
        'hiring_type': data.get('hiring_type', [{}])[0].get('data', ''),
        'tentative_start': data.get('tentative_start', [{}])[0].get('data', ''),
        'secondary_skills': [skill.get('skill', '') for skill in data.get('secondary_skills', [])],
        'primary_skills': [skill.get('skill', '') for skill in data.get('primary_skills', [])],
        'working_time_zone': data.get('working_time_zone', [{}])[0].get('data', ''),
        'tools_used': [tool.get('data', '') for tool in data.get('tools_used', [])],
        'no_of_rounds': data.get('no_of_rounds', ''),
        'interview_rounds': [{'round_number': round_info.get('round_number', ''), 'round_name': round_info.get('round_name', '')} for round_info in data.get('interview_rounds', [])],
        'communication_skill': data.get('communication_skill', [{}])[0].get('data', '')
    }

    return project_details



# async def questionAnswerArray(project_info):
#     # manual_questions = project_info.get('data', {}).get('ai_interviewer', {}).get('manual', {}).get('questions', [])
#     # technical_questions = project_info.get('data', {}).get('ai_interviewer', {}).get('technical', {}).get('questions', [])
#     global current_question_index
#     global temp_interview_questions
#     # manual_questions = project_info.get('data', {}).get('interview', {}).get('question_format', {}).get('manual', {}).get('questions', [])
#     communication_questions = project_info.get('data', {}).get('interview', {}).get('question_format', {}).get('communication', {}).get('questions', [])
    
#     technical_questions = project_info.get('data', {}).get('interview', {}).get('question_format', {}).get('technical', {}).get('questions', [])
#     temp_interview_questions = [q['question'] for q in  technical_questions +communication_questions]
#     question = {
#                 'index': current_question_index,
#                 'question': temp_interview_questions[current_question_index],
#                 'temp_interview_questions':temp_interview_questions
#             }
#     return temp_interview_questions

async def questionAnswerArray(project_info):
    global current_question_index
    global temp_interview_questions

    # Retrieve questions from the project_info dictionary
    communication_questions = project_info.get('data', {}).get('ai_interviewer', {}).get('communication', {}).get('questions', [])
    technical_questions = project_info.get('data', {}).get('ai_interviewer', {}).get('technical', {}).get('questions', [])

    # Combine the questions from both categories
    temp_interview_questions = [q['question'] for q in technical_questions + communication_questions]

    # Prepare the question dictionary for the current index
    question = {
        'index': current_question_index,
        'question': temp_interview_questions[current_question_index],
        'temp_interview_questions': temp_interview_questions
    }

    # Print for debugging (optional)


    return temp_interview_questions







async def get_Interview_Question(projectId):
    # Encode the job_id in base64
    async with httpx.AsyncClient() as client:

        url = f"{job_management_service_v2_url}{projectId}"

        # print('url',url)
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def get_project_info(projectId):
    # Encode the job_id in base64
    async with httpx.AsyncClient() as client:
    
    
        url = f"{job_management_service_v1_url}{projectId}"

       
        response = await client.get(url)
        response.raise_for_status()
        return response.json()



# def chain_general():
#     # Replace with your actual chain_general initialization
#     return SimpleChain(ChatPromptTemplate.from_template("General prompt"), llm, output_parser)

    

# @staticmethod
# async def clean_audio(audio_numpy, sr):
#     b, a = signal.butter(4, 100/(0.5*sr), btype='high')
#     audio_numpy = signal.filtfilt(b, a, audio_numpy)
#     audio_numpy, _ = librosa.effects.trim(audio_numpy, top_db=20)
#     return audio_numpy
# async def transcribe_audio(self, blob_data, send_partial=False):
#     async def process_audio_segment(blob_data):
#         audio_segment = AudioSegment.from_file(BytesIO(blob_data), format="wav")
#         temp_wav_file = "temp_audio.wav"
#         audio_segment.export(temp_wav_file, format="wav")
#         return temp_wav_file
#     temp_wav_file = await asyncio.to_thread(process_audio_segment, blob_data)
#     try:
#         audio_numpy, sr = librosa.load(temp_wav_file, sr=None)
#         audio_numpy = await self.clean_audio(audio_numpy, sr)
#         if sr != 16000:
#             audio_numpy = librosa.resample(audio_numpy, orig_sr=sr, target_sr=16000)
#         audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))
#         input_values = self.tokenizer(audio_numpy, return_tensors="pt", padding=True).input_values
#         with torch.no_grad():
#             logits = self.wav2vec2_model(input_values).logits
#         predicted_ids = torch.argmax(logits, dim=-1)
#         transcription = self.tokenizer.batch_decode(predicted_ids)[0].lower()
#     except Exception as e:
#         print(f"Error during audio processing or transcription: {e}")
#         return ""
#     if send_partial:
#         self.ongoing_transcription += transcription
#         return self.ongoing_transcription
#     else:
#         complete_sentence = self.ongoing_transcription + transcription
#         self.ongoing_transcription = ""
#         return complete_sentence

def summarize_text(text: str) -> str:
    # Ensure the text does not exceed model's max length
    # max_length = 50  # Set appropriate max length based on model
    # if len(text) > max_length:
        # Summarize the text
    summary = summarizer(text, max_length=50, min_length=30, do_sample=False)
    return summary[0]['summary_text']
    # return text