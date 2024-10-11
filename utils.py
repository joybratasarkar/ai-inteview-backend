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
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


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
move_on_responses = [
    "next question please", "I don’t know", "pass", "skip", "move on", 
    "end the interview", "let's stop here", "no more questions", "I’m done", 
    "that’s all", "let's wrap up", "I’m finished"
]
move_on_embeddings = model.encode(move_on_responses, convert_to_tensor=True)
summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


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
    






# AgentState Class to track state information during the interview
class AgentState:
    def __init__(self, summary=None, last_answer=None):
        self.summary = summary
        self.last_answer = last_answer
        self.asked_questions = []  # Keep a history of asked questions
        self.last_question = None  # Track the last question asked
        self.question_count = 0    # Track the number of questions asked

# Parse the PDF resume
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

# Summarize the resume
async def summarize_resume(text: str, min_sentences=6, max_sentences=15):
    if not text:
        raise ValueError("Empty text received for summarization.")

    start_time = time.time()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count=max_sentences)
    selected_summary = summary_sentences[:max(min_sentences, len(summary_sentences))]
    summary = " ".join(str(sentence) for sentence in selected_summary)
    
    end_time = time.time()
    logging.info(f"Summarization took {end_time - start_time:.2f} seconds.")
    
    return summary 

# Generate initial question
def generate_initial_question(summary: str) -> str:
    logging.info(f"Generating initial question based on summary: {summary}")
    prompt = f"""
    Based on the following candidate summary:
    {summary}
    Generate a first interview question that covers both the candidate's skills and their most recent work experience.
    """
    response = llm.invoke(prompt)
    return response.content.strip()

# Determine if user intent indicates moving on
# Updated `is_move_on_intent` with refined matching and more diversified phrases
def is_move_on_intent(answer: str, threshold=0.3) -> (bool, str):
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    
    # Calculate cosine similarities between the answer and each move-on response
    similarities = util.pytorch_cos_sim(answer_embedding, move_on_embeddings).squeeze(0)
    max_similarity, best_match_index = similarities.max().item(), similarities.argmax().item()
    
    # Retrieve the best matching phrase
    best_match_phrase = move_on_responses[best_match_index]
    logging.info(f"Best match phrase: '{best_match_phrase}' with similarity score: {max_similarity}")

    # Determine if this similarity exceeds the threshold
    return max_similarity > threshold, best_match_phrase if max_similarity > threshold else None



# Generate follow-up question based on intent and simulate realistic interview behavior
def generate_follow_up_question(answer: str, summary: str, question: str, asked_questions: list) -> str:
    # Check if the candidate's response matches a move-on intent and get the matched phrase
    intent_score, best_match_phrase = is_move_on_intent(answer)
    logging.info(f"Best match phrase: '{best_match_phrase}' with similarity score: {intent_score}")

    # Route the prompt based on the best match phrase
    if best_match_phrase in ["I don’t know", "pass", "skip"]:
        # If the candidate doesn't know or wants to skip, move to a different topic
        prompt = (
            f"The candidate's response of '{answer}' indicates uncertainty or a desire to skip. "
            f"As an interviewer, ask a question that explores another aspect of their role related to '{summary}', "
            f"and avoid the previous topic '{question}'. Focus on a new area of expertise."
        )
    elif best_match_phrase in ["next question please", "move on"]:
        # Candidate wants to move on, proceed to the next relevant topic
        prompt = (
            f"The candidate's response of '{answer}' suggests they want to proceed to another topic. "
            f"Ask them to describe their experience with a different technology or project related to their role, "
            f"focusing on skills listed in their resume summary: '{summary}'."
        )
    elif best_match_phrase in ["end the interview", "let's stop here", "no more questions", "I’m done", "that’s all", "let's wrap up", "I’m finished"]:
        # Candidate wants to end the interview
        logging.info("Ending interview based on candidate's response.")
        return "Thank you for your time. We’ll conclude the interview here. Best of luck with your next steps!"
    else:
        # Standard case for follow-up if no specific move-on intent is detected
        prompt = (
            f"Candidate's answer: '{answer}'. Resume summary: '{summary}'. Previous question: '{question}'. "
            f"Generate a follow-up question that dives into technical specifics, focusing on their usage of "
            f"relevant tools, techniques, or technologies, and ask them to elaborate on the process."
        )

    try:
        response = llm.invoke(prompt)
        next_question = response.content.strip()
        
        # Log the generated follow-up question
        logging.info(f"Generated follow-up question: {next_question}")

        # Check for previously asked questions to avoid repetition
        if next_question in asked_questions:
            prompt += " Ensure this follow-up question is unique and hasn't been covered already."
            response = llm.invoke(prompt)
            next_question = response.content.strip()
            logging.info(f"Modified follow-up question to avoid repetition: {next_question}")

        return next_question

    except Exception as e:
        logging.error(f"Error in generate_follow_up_question: {e}")
        return "Could you clarify your answer or provide more details about your interest?"




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
        if self.state.question_count >= 4:  # Limit questions to a maximum of 8
            return self.end_interview()
        first_question = generate_initial_question(self.state.summary)
        self.state.asked_questions.append(first_question)
        self.state.last_question = first_question
        self.state.question_count += 1
        return first_question

    def follow_up_question(self):
        answer = self.state.last_answer
        summary = self.state.summary
        question = self.state.last_question

        if self.state.question_count >= 4:  # Limit to a maximum of 8 questions
            return self.end_interview()

        next_question = generate_follow_up_question(answer, summary, question, self.state.asked_questions)
        if next_question not in self.state.asked_questions:
            self.state.asked_questions.append(next_question)
            self.state.last_question = next_question
            self.state.question_count += 1
        return next_question

    def end_interview(self):
        logging.info("Interview ended.")
        return "Thank you for the interview!"