from pydub import AudioSegment, silence
import numpy as np
from transformers import pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import pdfplumber
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import logging
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()

# Initialize models
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.7)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define response categories and combine all responses into a master list
response_categories = {
    "Move On": ["next question please", "pass", "skip", "move on", "let's move ahead", "can we proceed?"],
    "Clarify": ["can you repeat that?", "could you clarify the question?", "what do you mean by that?", "please explain"],
    "Unsure": ["I'm not sure", "I don’t know", "I'm uncertain about this", "I can't give a definite answer"],
    "Finished": ["that's all I can say", "I think I’ve covered it", "I believe I've answered the question"],
    "End Interview": ["end the interview", "I’m done with the session", "no more questions", "let's wrap up"],
    "General": ["I see", "interesting", "okay", "got it", "sounds good"],
    "Positive": ["I'm excited about this", "this sounds interesting", "I would love to discuss this more"],
    "Negative": ["this doesn't interest me", "I find this irrelevant", "I'm not interested in this topic"],
    "Question": ["can I ask a question?", "what about this aspect?", "how does this work?", "what do you think?"],
}

all_responses = [(response, label) for label, responses in response_categories.items() for response in responses]

class AgentState:
    def __init__(self, summary=None, max_questions=8):
        self.summary = summary
        self.last_answer = None
        self.asked_questions = []
        self.last_question = None
        self.question_count = 0
        self.max_questions = 4  # Dynamic max questions

# Utility Functions

def process_audio_blob(audio_blob, chunk_size=1024):
    """
    Process large audio blobs in chunks to avoid memory overload.
    """
    try:
        full_audio_segment = AudioSegment.empty()
        total_chunks = len(audio_blob) // chunk_size

        for i in range(total_chunks + 1):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio_blob))
            chunk = audio_blob[start:end]

            if chunk:
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                audio_segment = AudioSegment(
                    audio_array.tobytes(),
                    frame_rate=16000,
                    sample_width=2,
                    channels=1
                )
                full_audio_segment += audio_segment

        return full_audio_segment

    except Exception as e:
        logging.error(f"Error processing audio blob: {e}")
        return None

def parse_resume(pdf_file_data: bytes):
    """
    Extract text from a PDF resume.
    """
    try:
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
        logging.error(f"Error parsing PDF: {str(e)}")
        raise ValueError(f"Failed to parse PDF: {str(e)}")

async def summarize_resume(text: str, min_sentences=6, max_sentences=15):
    """
    Summarize the resume text using LexRankSummarizer.
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count=max_sentences)
        selected_summary = summary_sentences[:max(min_sentences, len(summary_sentences))]
        summary = " ".join(str(sentence) for sentence in selected_summary)
        return summary

    except Exception as e:
        logging.error(f"Error summarizing resume: {e}")
        return "Error in summarizing the resume."

def dynamic_rate_sentence(sentence):
    """
    Dynamically rate user input and detect intent using cosine similarity.
    """
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    response_embeddings = model.encode([resp[0] for resp in all_responses], convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(sentence_embedding, response_embeddings).squeeze(0)

    max_similarity, best_match_index = similarities.max().item(), similarities.argmax().item()
    best_match, detected_intent = all_responses[best_match_index]

    normalized_score = (max_similarity + 1) / 2

    return normalized_score, best_match, detected_intent

def get_prompt_for_intent(intent, answer, summary, question):
    """
    Returns the appropriate prompt based on the detected intent.
    """
    prompts = {
        "Move On": (
            f"The candidate wants to move forward, responding with: '{answer}'. "
            f"Ask a new question related to their skills or experience, avoiding the topic from: '{question}'."
        ),
        "Clarify": (
            f"The candidate asked for clarification: '{answer}'. Rephrase the question: '{question}', "
            f"keeping it related to '{summary}'."
        ),
        "Unsure": (
            f"The candidate was unsure: '{answer}'. Ask a simpler question exploring another aspect of '{summary}'."
        ),
        "Finished": (
            f"The candidate indicated completion: '{answer}'. Ask a follow-up question focusing on a different topic."
        ),
        "End Interview": (
            f"The candidate wants to conclude: '{answer}'. Prepare a closing statement."
        ),
        "General": (
            f"The candidate gave a general response: '{answer}'. Ask a more specific follow-up question."
        ),
        "Question": (
            f"The candidate asked a question: '{answer}'. Respond and ask a related follow-up."
        ),
    }
    return prompts.get(intent, (
        f"Candidate's response: '{answer}'. Resume summary: '{summary}'. Previous question: '{question}'. "
        f"Generate a detailed follow-up question focusing on technical skills or relevant experiences."
    ))
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

def invoke_llm_with_retry(prompt, asked_questions, max_retries=2):
    """
    Invokes the language model to generate a follow-up question and ensures it's unique.
    """
    for _ in range(max_retries):
        try:
            response = llm.invoke(prompt)
            next_question = response.content.strip()

            if next_question not in asked_questions:
                asked_questions.append(next_question)
                logging.info(f"Generated follow-up question: {next_question}")
                return next_question

            logging.info(f"Duplicate question detected. Retrying...")
            prompt += " Ensure this question is unique."

        except Exception as e:
            logging.error(f"Error generating follow-up question: {e}")

    return "Could you clarify your answer or provide more details?"

def generate_initial_question(summary: str) -> str:
    """
    Generate the first interview question based on the summarized resume.
    """
    prompt = f"Based on the candidate's resume summary: '{summary}', generate an initial interview question."
    try:
        response = llm.invoke(prompt)
        initial_question = response.content.strip()
        return initial_question

    except Exception as e:
        logging.error(f"Error generating initial question: {e}")
        return "Could you describe your recent experience?"

def generate_follow_up_question(answer: str, summary: str, question: str, asked_questions: list) -> str:
    """
    Generate follow-up questions based on detected intent and previous responses.
    """
    rating, best_match_phrase, detected_intent = dynamic_rate_sentence(answer)
    logging.info(f"User's response: {answer}, Detected intent: {detected_intent}")

    prompt = get_prompt_for_intent(detected_intent, answer, summary, question)
    logging.info(f"Generated prompt: {prompt}")

    next_question = invoke_llm_with_retry(prompt, asked_questions)
    return next_question

# InterviewGraph for managing interview flow
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
        if self.state.question_count >= self.state.max_questions:
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

        if self.state.question_count >= self.state.max_questions:
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
