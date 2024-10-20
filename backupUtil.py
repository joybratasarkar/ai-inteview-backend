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
import time
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()

# Initialize models
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.7)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

classifier = pipeline("zero-shot-classification", model="facebook/bart-base")
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
class AgentState:
    def __init__(self, summary=None, last_answer=None):
        self.summary = summary
        self.last_answer = last_answer
        self.asked_questions = []  # Keep a history of asked questions
        self.last_question = None  # Track the last question asked
        self.question_count = 0    # Track the number of questions asked
# Define categorized responses for intents
# response_categories = {
#     "Move On": ["next question please", "pass", "skip", "move on"],
#     "Clarify": ["can you repeat that?", "could you clarify the question?"],
#     "Unsure": ["I'm not sure", "I don’t know"],
#     "Finished": ["that's all I can say", "I think I’ve covered it"],
# }
response_categories = {
    # Move On / Skip Responses
    "Move On": [
        "next question please", "pass", "skip", "move on",
        "can we go to the next question?", "I'd like to skip this",
        "I'm done with this topic", "let's move ahead",
        "let's change the topic", "I'd like to move forward",
        "can we proceed?", "I want to skip this",
        "can we jump to the next question?", "I'm ready to move on",
        "let’s keep going", "let’s not dwell on this",
        "I’ve answered enough on this", "no further comments",
        "I'd prefer to pass on this"
    ],
    
    # Clarify / Repeat Requests
    "Clarify": [
        "can you repeat that?", "could you clarify the question?",
        "can you explain that differently?", "what do you mean by that?",
        "could you give an example?", "could you elaborate on that?",
        "I'm not clear on this question", "can you rephrase that?",
        "I didn’t catch that, can you say it again?", "what exactly do you mean?",
        "can you break that down?", "could you make it simpler?"
    ],
    
    # Unsure / Unprepared Responses
    "Unsure": [
        "I'm not sure", "I don’t know", "I'm not prepared for this question",
        "this is outside my knowledge", "I can't give a definite answer",
        "I haven’t thought about this before", "I'm uncertain about this",
        "I don’t have enough information", "I’m not confident with this",
        "I’m not familiar with this topic", "I need more context to answer"
    ],
    
    # Finished / Satisfied Responses
    "Finished": [
        "that's all I can say", "I think I’ve covered it",
        "I believe I've answered the question", "that's my final answer",
        "that's all from my side", "I have nothing else to add",
        "I’m satisfied with my answer", "I’ve provided enough detail",
        "I think this topic is clear now", "I don’t have more to share on this",
        "I think we’ve covered everything", "I’m done discussing this"
    ],
    
    # Agree Responses
    "Agree": [
        "I agree with that", "that makes sense", "I think you're right",
        "I support that idea", "I can see your point",
        "I’m on board with that", "I think we’re aligned",
        "I completely agree", "I have no objections",
        "I see your perspective and agree"
    ],
    
    # Disagree Responses
    "Disagree": [
        "I disagree with that", "I'm not convinced by that",
        "I have a different opinion", "I see it differently",
        "I have another perspective", "I don't think that's accurate",
        "I have some concerns about that", "I'm not aligned with this idea",
        "I see your point, but I disagree", "I don't share the same view"
    ],
    
    # Positive / Enthusiastic Responses
    "Positive": [
        "I'm excited about this", "this sounds interesting",
        "I would love to discuss this more", "this aligns with my experience",
        "I'm eager to explore this further", "I’m enthusiastic about this topic",
        "this fits well with my skills", "this is a strong area for me",
        "I find this topic fascinating", "I’m happy to delve deeper",
        "this is something I enjoy", "I’m passionate about this area"
    ],
    
    # Negative / Disinterested Responses
    "Negative": [
        "this doesn't interest me", "I'm not excited about this",
        "this doesn't align with my skills", "I find this irrelevant",
        "I'm not interested in this topic", "this isn't appealing to me",
        "I don’t enjoy this topic", "this isn't my area of interest",
        "I have no enthusiasm for this", "I’d rather focus on something else",
        "I don’t see the relevance of this question"
    ],
    
    # End Interview / Conclude
    "End Interview": [
        "end the interview", "I think we’re done here",
        "that’s it for today", "I’m finished with the interview",
        "I have no further questions", "can we conclude now?",
        "I’d like to wrap up", "this is the end of the interview for me",
        "I’m ready to end the discussion", "let's finish up",
        "can we call it a day?", "I’m done with the session",
        "no more questions from my side", "I’d like to end this interview"
    ],
    
    # General / Neutral Responses
    "General": [
        "I see", "interesting", "okay", "I understand",
        "got it", "noted", "all right", "fair enough",
        "let’s proceed", "sure", "I hear you", 
        "that's clear", "makes sense", "I see where you're coming from",
        "understood", "sounds good"
    ],
    
    # Questions / Counter-Questions
    "Question": [
        "can I ask a question?", "what about this aspect?",
        "how does this work?", "could you tell me more about this?",
        "what do you think?", "is there more to this?",
        "could you expand on this?", "what’s your perspective?",
        "what are your thoughts on this?", "how do you view this?"
    ]
}


# Combine all responses into a master list with labels
all_responses = [(response, label)
                 for label, responses in response_categories.items()
                 for response in responses]
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
        logging.error(f"Error processing audio blob in chunks: {e}")
        return None

def clear_audio_segment(audio_segment, silent_ranges):
    """
    Removes the detected silence from the audio segment.
    """
    if not silent_ranges:
        return audio_segment

    cleaned_audio = AudioSegment.empty()
    prev_end = 0

    for start, end in silent_ranges:
        cleaned_audio += audio_segment[prev_end:start]
        prev_end = end

    cleaned_audio += audio_segment[prev_end:]
    return cleaned_audio

def detect_silence(audio_segment, min_silence_len=5000):
    """
    Detects and removes silence from the audio segment.
    """
    try:
        chunk_size = 5000  # 5 seconds
        overall_dBFS = int(audio_segment.dBFS)
        silence_thresh = -30

        silent_ranges = []

        for start in range(0, len(audio_segment), chunk_size):
            end = min(start + chunk_size, len(audio_segment))
            chunk = audio_segment[start:end]

            ranges = silence.detect_silence(
                chunk,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            for silence_start, silence_end in ranges:
                silent_ranges.append((start + silence_start, start + silence_end))

        if silent_ranges:
            cleaned_audio = clear_audio_segment(audio_segment, silent_ranges)
            return True, cleaned_audio, overall_dBFS

        return False, audio_segment, overall_dBFS

    except Exception as e:
        logging.error(f"Error detecting or clearing silence: {e}")
        return False, audio_segment, int(audio_segment.dBFS)



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
        logging.error(f"Error in summarizing resume: {e}")
        return "Error in summarizing the resume."

# Function to rate user input and dynamically detect intent
def dynamic_rate_sentence(sentence):
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    response_embeddings = model.encode([resp[0] for resp in all_responses], convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = util.pytorch_cos_sim(sentence_embedding, response_embeddings).squeeze(0)

    # Get maximum similarity score and the best matching response
    max_similarity, best_match_index = similarities.max().item(), similarities.argmax().item()

    best_match, detected_intent = all_responses[best_match_index]

    # Normalize the similarity score
    normalized_score = (max_similarity + 1) / 2

    return normalized_score, best_match, detected_intent


def generate_follow_up_question(answer: str, summary: str, question: str, asked_questions: list) -> str:
    """
    Generate follow-up questions based on detected intent and previous responses.
    """
    # Detect intent and log it
    rating, best_match_phrase, detected_intent = dynamic_rate_sentence(answer)
    logging.info(f"User's response: {answer}")
    logging.info(f"Detected intent: {detected_intent}")

    # Create the prompt based on detected intent
    if detected_intent == "Move On":
        prompt = (
            f"The candidate wants to move forward, responding with: '{answer}'. "
            f"Ask a new question related to their skills or experience as per the summary: '{summary}', "
            f"avoiding the topic from the previous question: '{question}'."
        )
    elif detected_intent == "Clarify":
        prompt = (
            f"The candidate asked for clarification with their response: '{answer}'. "
            f"Rephrase the previous question: '{question}' in a clearer and simpler way, "
            f"while keeping it relevant to their resume summary: '{summary}'."
        )
    elif detected_intent == "Unsure":
        prompt = (
            f"The candidate was unsure with their response: '{answer}'. "
            f"Ask a simpler, more fundamental question that explores another aspect of their skills or experience, "
            f"as outlined in the summary: '{summary}'."
        )
    elif detected_intent == "Finished":
        prompt = (
            f"The candidate indicated completion with their response: '{answer}'. "
            f"Ask a follow-up question focusing on a different topic from their summary: '{summary}', "
            f"encouraging them to share new insights or experiences."
        )
    elif detected_intent == "Agree":
        prompt = (
            f"The candidate agreed with the discussion, saying: '{answer}'. "
            f"Ask a follow-up question that dives deeper into their experience related to the summary: '{summary}', "
            f"focusing on specific projects or skills they are confident in."
        )
    elif detected_intent == "Disagree":
        prompt = (
            f"The candidate disagreed with the discussion, responding with: '{answer}'. "
            f"Ask a question that allows them to share an alternative viewpoint, "
            f"keeping it aligned with the summary: '{summary}'."
        )
    elif detected_intent == "Positive":
        prompt = (
            f"The candidate showed enthusiasm, saying: '{answer}'. "
            f"Ask a more detailed question related to the area they are passionate about, "
            f"as mentioned in the summary: '{summary}'."
        )
    elif detected_intent == "Negative":
        prompt = (
            f"The candidate expressed disinterest with their response: '{answer}'. "
            f"Shift the discussion to a different topic from their summary: '{summary}', "
            f"and ask a question that aligns with their areas of interest."
        )
    elif detected_intent == "End Interview":
        prompt = (
            f"The candidate wants to conclude the interview, stating: '{answer}'. "
            f"Prepare a concluding statement and wrap up the interview with final thoughts."
        )
    elif detected_intent == "General":
        prompt = (
            f"The candidate gave a general response: '{answer}'. "
            f"Ask a more specific follow-up question that delves into their experience, "
            f"as outlined in the summary: '{summary}'."
        )
    elif detected_intent == "Question":
        prompt = (
            f"The candidate asked a counter-question: '{answer}'. "
            f"Respond to their inquiry and then ask a related follow-up question based on their summary: '{summary}'."
        )
    else:
        prompt = (
            f"Candidate's response: '{answer}'. Resume summary: '{summary}'. Previous question: '{question}'. "
            f"Generate a detailed follow-up question focusing on technical skills or relevant experiences."
        )

    # Log the generated prompt
    logging.info(f"Generated prompt: {prompt}")

    try:
        # Use the language model to generate the follow-up question
        response = llm.invoke(prompt)
        next_question = response.content.strip()

        # Ensure the follow-up question is unique
        if next_question in asked_questions:
            prompt += " Ensure this question is unique."
            response = llm.invoke(prompt)
            next_question = response.content.strip()

        # Log the generated follow-up question
        logging.info(f"Generated follow-up question: {next_question}")
        asked_questions.append(next_question)
        return next_question

    except Exception as e:
        logging.error(f"Error in generate_follow_up_question: {e}")
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
        logging.error(f"Error in generating initial question: {e}")
        return "Could you describe your recent experience?"
        

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