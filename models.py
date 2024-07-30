# models.py

# Initialize variables
sessions = {}
interview_questions = []
current_question_index = 0
db = None
temp_interview_questions:[]
chain_general:None
general_prompt:None
vector_store = None
segmented_texts = None
candidate_responses = {}


# # Function to detect silence in audio
# def detect_silence(audio_blob):
#     audio = AudioSegment.from_file(io.BytesIO(audio_blob), format="webm")
#     silence_threshold = -30.0  # in dB
#     silence_duration = 4000  # in milliseconds (4 seconds)

#     silence_chunks = silence.detect_silence(audio, min_silence_len=silence_duration, silence_thresh=silence_threshold)
#     if silence_chunks:
#         print("Detected silence.")
#         return True
#     else:
#         print("No silence detected.")
#         return False
