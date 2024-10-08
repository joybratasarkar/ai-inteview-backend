from pydub import AudioSegment, silence
import numpy as np
from transformers import pipeline
import httpx
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import scipy.signal as signal
import torch
import asyncio
import pdfplumber
from io import BytesIO
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
import logging

load_dotenv()

current_question_index = 0
temp_interview_questions:[]
# output_parser = StrOutputParser()


openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.7)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

classifier = pipeline("zero-shot-classification", model="facebook/bart-base")
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Precompute embeddings for "move on" intent responses
move_on_responses = ["next question please", "I don’t know", "pass", "skip", "move on",]
move_on_embeddings = model.encode(move_on_responses, convert_to_tensor=True)


def process_audio_blob(audio_blob, chunk_size=1024):
    """
    Process large audio blobs in chunks to avoid memory overload.
    
    Args:
    - audio_blob (bytes): Raw audio data (assumed to be PCM 16-bit mono).
    - chunk_size (int): Size of each chunk in bytes for processing.

    Returns:
    - AudioSegment: Combined AudioSegment of all processed chunks, or None if an error occurs.
    """
    try:
        # Initialize an empty AudioSegment to hold the processed chunks
        # full_audio_segment = AudioSegment.silent(duration=0, frame_rate=16000)
        full_audio_segment = AudioSegment.empty()

        # Calculate the total number of chunks
        total_chunks = len(audio_blob) // chunk_size

        # Process each chunk
        for i in range(total_chunks + 1):
            # Extract the current chunk from the audio blob
            start = i * chunk_size
            end = min(start + chunk_size, len(audio_blob))
            chunk = audio_blob[start:end]

            if chunk:
                # Convert the chunk to a numpy array (assuming 16-bit PCM)
                audio_array = np.frombuffer(chunk, dtype=np.int16)

                # Create an AudioSegment from the numpy array
                audio_segment = AudioSegment(
                    audio_array.tobytes(),
                    frame_rate=16000,  # Assuming 16kHz frame rate for the audio
                    sample_width=2,    # 2 bytes (16-bit) per sample
                    channels=1         # Mono audio
                )

                # Concatenate the chunk to the full audio
                full_audio_segment += audio_segment

        return full_audio_segment

    except Exception as e:
        logger.error(f"Error processing audio blob in chunks: {e}")
        return None

def clear_audio_segment(audio_segment, silent_ranges):
    """
    Removes the detected silence from the audio segment.

    Args:
        audio_segment: The audio segment to clean.
        silent_ranges: The list of tuples (start, end) representing silence ranges.

    Returns:
        AudioSegment: The cleaned audio segment without silence.
    """
    # If there are no silent ranges, return the original audio
    if not silent_ranges:
        return audio_segment

    # Create a new segment by removing all silent ranges
    cleaned_audio = AudioSegment.empty()
    prev_end = 0

    # Go through the detected silent ranges and exclude them
    for start, end in silent_ranges:
        # Append the non-silent part to cleaned_audio
        cleaned_audio += audio_segment[prev_end:start]
        prev_end = end

    # Append the remaining non-silent part after the last silent range
    cleaned_audio += audio_segment[prev_end:]

    return cleaned_audio




def detect_silence(audio_segment, min_silence_len=5000):
    """
    Detects if there's any silence in the audio segment that is at least `min_silence_len` milliseconds long
    and removes that silence.

    Args:
        audio_segment: The audio segment to check for silence.
        min_silence_len: Minimum length of silence in milliseconds to detect (default 5000ms = 5 seconds).

    Returns:
        tuple: (bool, AudioSegment, int) where bool is True if silence of specified length is detected,
               AudioSegment is the cleaned audio without silence, and int is the overall dBFS.
    """
    try:
        # Set chunk size to 5 seconds for more granular processing
        chunk_size = 5000  # 5 seconds

        # Calculate the overall dBFS for the entire audio segment
        overall_dBFS = audio_segment.dBFS
        overall_dBFS_int = int(overall_dBFS)  # Convert dBFS to an integer for simplicity

        print(f"Overall dBFS: {overall_dBFS} dBFS")

        # Set a fixed threshold to detect silence
        silence_thresh = -30  # dBFS threshold for detecting silence

        silent_ranges = []

        # Iterate over audio in chunks to detect silence
        for start in range(0, len(audio_segment), chunk_size):
            end = min(start + chunk_size, len(audio_segment))
            chunk = audio_segment[start:end]
            
            # Detect silence in the chunk
            ranges = silence.detect_silence(
                chunk,
                min_silence_len=min_silence_len,  # Minimum silence duration to detect
                silence_thresh=silence_thresh      # Threshold for detecting silence
            )
            
            # Append detected silent ranges from this chunk to the overall list
            for silence_start, silence_end in ranges:
                silent_ranges.append((start + silence_start, start + silence_end))

        # If silence was detected, clean the audio
        if silent_ranges:
            cleaned_audio = clear_audio_segment(audio_segment, silent_ranges)
            return True, cleaned_audio, overall_dBFS_int

        # No silence detected, return the original audio
        return False, audio_segment, overall_dBFS_int

    except Exception as e:
        print(f"Error detecting or clearing silence: {e}")
        return False, audio_segment, overall_dBFS_int
 

def is_general_question(question):
    # candidate_labels = ['general', 'question', 'answer','no-answer','dont-know-the-answer','move-to-next-question']
    candidate_labels = ['general', 'question', 'answer','Unanswered']

    
    result = classifier(question, candidate_labels=candidate_labels, multi_label=True,device=0)
    print('Classifier result:', result) 
    
    scores = {label: score for label, score in zip(result['labels'], result['scores'])} # Get the scores for the labels

    return  scores
    








class AgentState:
    def __init__(self, summary=None, last_answer=None):
        self.summary = summary
        self.last_answer = None
        self.asked_questions = []  # Keep a history of asked questions
        self.last_question = None  # Track the last question asked

# Function to parse PDF resume using pdfplumber and return text
def parse_resume(pdf_file_data: bytes):
    try:
        logging.info("Attempting to parse PDF resume")
        text = ""
        pdf_file = BytesIO(pdf_file_data)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text:
            raise ValueError("No text found in the PDF.")
        logging.info(f"Parsed text length: {len(text)} characters")
        return text
    except Exception as e:
        logging.error(f"Error parsing file: {str(e)}")
        raise ValueError(f"Failed to parse file: {str(e)}")

# Summarize the resume using LangChain's summarization chain
def summarize_resume(text: str, method="map_reduce"):
    documents = [Document(page_content=text)]
    summarize_chain = load_summarize_chain(llm, chain_type=method)
    summary = summarize_chain.invoke(documents)
    return summary

# Generate the initial interview question based on the resume summary
def generate_initial_question(summary: str) -> str:
    logging.info(f"Generating initial question based on summary: {summary}")
    prompt = f"""
    Based on the following candidate summary:
    {summary}
    Generate a first interview question that covers both the candidate's skills and their most recent work experience.
    """
    response = llm.invoke(prompt)
    return response.content.strip()

# Detect if the candidate's response implies they want to move on
# def is_move_on_intent(answer: str) -> bool:
#     answer_embedding = model.encode(answer, convert_to_tensor=True)
#     similarities = util.pytorch_cos_sim(answer_embedding, move_on_embeddings)
#     return max(similarities[0]).item() > 0.7
def is_move_on_intent(answer: str) -> bool:
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(answer_embedding, move_on_embeddings)
    
    # Log the similarity scores for debugging
    similarity_scores = similarities[0].cpu().numpy()
    logging.info(f"Similarity scores: {similarity_scores}")
    
    # Determine if any of the similarities exceed the threshold
    max_similarity = max(similarity_scores)
    logging.info(f"Max similarity score: {max_similarity}")
    
    return max_similarity > 0.7  # Adjust the threshold as needed

# Generate follow-up questions based on the candidate's answer and the resume summary
def generate_follow_up_question(answer: str, summary: str, question: str, asked_questions: list) -> str:
    if is_move_on_intent(answer):
        prompt = f"Based on the following resume summary: {summary}. Generate the next interview question."
        response = llm.invoke(prompt)
        next_question = response.content.strip()
        return next_question

    prompt = f"Candidate's answer: {answer}. Resume summary: {summary}. Generate a follow-up interview question."
    response = llm.invoke(prompt)
    return response.content.strip()

# StateGraph for managing interview process
class InterviewGraph(StateGraph):
    def __init__(self, state: AgentState):
        super().__init__(state)
        self.state = state

        self.add_node("START_INTERVIEW", self.start_interview)
        self.add_node("ASK_QUESTION", self.ask_question)
        self.add_node("FOLLOW_UP", self.follow_up_question)
        self.add_node("END_INTERVIEW", self.end_interview)

        self.add_edge("__start__", "START_INTERVIEW")
        self.add_edge("START_INTERVIEW", "ASK_QUESTION")
        self.add_edge("ASK_QUESTION", "FOLLOW_UP")
        self.add_edge("FOLLOW_UP", "ASK_QUESTION")
        self.add_edge("FOLLOW_UP", "END_INTERVIEW")

    def start_interview(self):
        logging.info("Interview started.")
        return "Let's start the interview!"

    def ask_question(self):
        first_question = generate_initial_question(self.state.summary)
        return first_question

    def follow_up_question(self):
        answer = self.state.last_answer
        summary = self.state.summary
        question = self.state.last_question
        return generate_follow_up_question(answer, summary, question, self.state.asked_questions)

    def end_interview(self):
        logging.info("Interview ended.")
        return "Thank you for the interview!"