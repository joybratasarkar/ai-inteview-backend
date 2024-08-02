from pydub import AudioSegment, silence
import numpy as np
from transformers import pipeline
import httpx
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import scipy.signal as signal
import torch
import asyncio
0

current_question_index = 0
temp_interview_questions:[]
# output_parser = StrOutputParser()


classifier = pipeline("zero-shot-classification", model="facebook/bart-base")
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

def process_audio_blob(audio_blob):
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
    candidate_labels = ['general', 'question', 'answer','Unanswered']

    
    result = classifier(question, candidate_labels=candidate_labels, multi_label=True,device=0)
    print('Classifier result:', result) 
    
    scores = {label: score for label, score in zip(result['labels'], result['scores'])} # Get the scores for the labels

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










async def get_project_info():

        return {
    "_id": "****************",
    "client_name": "infinix",
    "client_id": "**********887",
    "client_poc": [
        {
            "_id": "**********",
            "email": "ioruw@sfa.zc",
            "mobile_number": "+919878799889",
            "client_poc": "sifusilfjsiol",
            "comment": "uiouiio"
        }
    ],
    "role": [
        {
            "_id": "****************",
            "role": "Blockchain Developer"
        }
    ],
    "experience_range": [
        {
            "_id": "**************",
            "data": "3-7"
        }
    ],
    "max_experience": 7,
    "min_experience": 3,
    "client_price": 333333,
    "ss_price": 333333,
    "month_of_engagement": [
        {
            "_id": "*********",
            "data": "3 Months"
        }
    ],
    "engagement_type": [
        {
            "_id": "*******",
            "data": "Full-Time Contract"
        }
    ],
    "no_requirements": 2,
    "tentative_start": [
        {
            "_id": "********",
            "data": "Not Specified"
        }
    ],
    "secondary_skills": [
        {
            "_id": "******",
            "skill": ".NET Compact Framework"
        }
    ],
    "primary_skills": [
        {
            "_id": "*******",
            "skill": "Typescript",
            "competency": [
                {
                    "_id": "******",
                    "data": "Expert"
                }
            ]
        },
        {
            "_id": "******",
            "skill": "Ruby on Rails",
            "competency": [
                {
                    "_id": "******",
                    "data": "Expert"
                }
            ]
        }
    ],
    "working_time_zone": [
        {
            "_id": "*****",
            "label": "Africa/Douala( UTC +00:13 )",
            "data": "Africa/Douala"
        }
    ],
    "working_hours": 40,
    "travel_preference": [
        {
            "_id": "*****",
            "data": "Remote"
        }
    ],
    "locations": [],
    "tools_used": [
        {
            "_id": "******",
            "data": "Agile Practice"
        }
    ],
    "system_provided": [
        {
            "_id": "******",
            "data": "Not Specified"
        }
    ],
    "no_of_rounds": 1,
    "interview_rounds": [
        {
            "round_number": 1,
            "round_name": "test"
        }
    ],
    "communication_skill": [
        {
            "_id": "*****",
            "data": "excellent"
        }
    ],
    "job_responsibility": "<p>test</p>",
    "interested_count": 0,
    "interviewing_count": 0,
    "hired_count": 0,
    "rejected_count": 0,
    "shortlisted_count": 0,
    "is_world_wide": true,
    "is_client_deleted": false,
    "job_platform": "******",
    "offer_sent_count": 0,
    "offer_reject_count": 0,
    "show_in_jd": false,
    "platform_type": "pre-hire",
    "type": "premium-job",
    "project_status": "draft",
    "rand_no": 64,
    "sales_poc": [
        {
            "team_member_id": "******",
            "first_name": "new presales",
            "last_name": "test",
            "profile_pic": null,
            "_id": "66139dc124dafe1e6601b3b7"
        }
    ],
    "status": "assigned",
    "mark_as_active": true,
    "mark_as_active_date": "2024-04-08T07:33:21.807Z",
    "is_screening_question_added": false,
    "linkedin_job_title": "SDE",
    "project_logs": [
        {
            "status": "added",
            "user_name": "admin admin",
            "user_id": "63c7f3183a3ac2508b654968",
            "user_role": "admin",
            "created_at": "2024-04-08T07:33:21.843Z",
            "_id": "66139dc124dafe1e6601b3b8"
        }
    ],
    "post_to_career_page": true,
    "department": [],
    "experience_level": [],
    "preffered_location": [],
    "education_type": [],
    "hiring_manager": [],
    "contact_person": [],
    "tags": [],
    "interview_rounds_ats": [],
    "cooling_period_department": [],
    "cooling_period_location": [],
    "internship_duration": [],
    "project_duration": [],
    "reason_for_job_close": [],
    "reason_for_job_paused": [],
    "talent_manager_associate": [
        {
            "talent_associate_id": "*****",
            "first_name": "vikrant",
            "last_name": "aswan",
            "email": "vikrant*****@yopmail.com",
            "createdAt": "2024-08-01T06:34:49.306Z",
            "_id": "66ab2c89cfb7f11877b4e145"
        }
    ],
    "vendor_details": [],
    "vendor_admin": [],
    "screening_question_added_by": [],
    "screening_question_updated_by": [],
    "ai_interviewer": [],
    "createdAt": "2024-04-08T07:33:21.854Z",
    "updatedAt": "2024-08-01T07:05:14.106Z",
    "project_id": "INF0001519",
    "__v": 0,
    "cron_date": "2024-04-18T00:00:00.000Z",
    "hiring_type": [
        {
            "_id": "6634b12b46acb34dc835f353",
            "data": "Contractual"
        }
    ],
    "dummy_no_requirements": 25,
    "ai_interviewer2": {
        "no_of_questions": 2,
        "technical": {
            "questions": [
                {
                    "question": "- Can you explain the difference between a module and a namespace in TypeScript?",
                    "_id": "66ab33aacfb7f11877b4e99c"
                },
                {
                    "question": "- How would you implement an authentication system in a Ruby on Rails application?",
                    "_id": "66ab33aacfb7f11877b4e99d"
                }
            ],
            "_id": "66ab33aacfb7f11877b4e99b"
        },
        "manual": {
            "questions": [],
            "keywords": []
        },
        "status": "active",
        "added_by_id": "*****",
        "added_by_name": "admin ***",
        "added_by_role": "admin",
        "_id": "***********",
        "createdAt": "2024-08-01T07:05:14.116Z"
    },
    "ai_interviewer_template_id": "*********",
    "screeningquestions": [],
    "no_requirements_fullfilled": 0,
    "job_counts": {
        "overall_counts": {
            "project_id": "66139dc124dafe1e6601b3b6",
            "shortlisted": "0",
            "clientsubmitted": "0",
            "selected": "0",
            "onboarded": "0",
            "interviewing": "0",
            "rejected": 0,
            "autoRejected": "1"
        },
        "applied": {
            "appliedDeveloper": 1,
            "vendorAppliedDeveloper": 0,
            "outboundDeveloper": 0,
            "recruiterFreelancerDeveloper": 0
        },
        "total_applied": 2,
        "shortListed": 0,
        "screening": 0,
        "vetting": {
            "skillVetting": 0,
            "verification": 0,
            "hrVetting": 0
        },
        "interviews": [
            {
                "round": 0
            },
            {
                "round": 0
            },
            {
                "round": 0
            },
            {
                "round": 0
            }
        ],
        "hired": {
            "selected": 0,
            "offerSent": 0
        },
        "clientOnboarding": 0,
        "rejected": {
            "movedToBenchpool": 0,
            "reject": 1
        },
        "clientSubmitted": 0
    }
}




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
