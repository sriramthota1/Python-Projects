from flask import Flask, request, jsonify, render_template
from transformers import pipeline, RagRetriever, RagSequenceForGeneration, RagTokenizer
from langdetect import detect, DetectorFactory
import torch
import os
from googletrans import Translator

# Fix random seed for reproducibility
DetectorFactory.seed = 0

# Initialize Flask app
app = Flask(__name__)


# Initialize the RAG model and retriever
def initialize_rag_model():
    try:
        rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", trust_remote_code=True)
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True,
                                                 trust_remote_code=True)
        rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever,
                                                             trust_remote_code=True)

        return retriever, rag_model, rag_tokenizer
    except Exception as e:
        print(f"Error initializing RAG model: {e}")
        return None, None, None


retriever, rag_model, rag_tokenizer = initialize_rag_model()

if not retriever or not rag_model or not rag_tokenizer:
    raise RuntimeError("Failed to load RAG model and components.")

# Initialize Google Translator
translator = Translator()


# Speech-to-Text Function with Language Detection
def speech_to_text_with_lang_detection(audio_file):
    """
    Convert speech to text using Whisper model and detect the language.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    try:
        result = asr(audio_file)
        text = result['text']
        detected_lang = detect(text)
        return text, detected_lang
    except Exception as e:
        print(f"Error in speech-to-text conversion: {e}")
        return "Error during transcription", "unknown"


# Translation Function using Google Translate API
def translate_text(text, src_lang, tgt_lang='en'):
    """
    Translate text from source language to target language using Google Translate API.
    """
    try:
        translated = translator.translate(text, src=src_lang, dest=tgt_lang)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# RAG Function
def generate_answer_with_rag(question):
    """
    Use RAG to generate an answer based on the input question.
    """
    inputs = rag_tokenizer(question, return_tensors="pt")
    try:
        with torch.no_grad():
            outputs = rag_model.generate(**inputs)
        answer = rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return answer
    except Exception as e:
        print(f"RAG generation error: {e}")
        return "Error generating answer with RAG."


# Process Audio Query Function
def process_audio_query(audio_file, target_lang='en'):
    """
    Process an audio file by detecting the language, transcribing it, translating it, and using RAG to generate an answer.
    """
    # Detect language and convert speech to text
    text, detected_lang = speech_to_text_with_lang_detection(audio_file)
    print(f"Detected language: {detected_lang}")
    print(f"Transcribed text: {text}")

    # Translate to English if needed
    if detected_lang != 'en' and detected_lang != 'unknown':
        translated_text = translate_text(text, src_lang=detected_lang, tgt_lang='en')
    else:
        translated_text = text

    # Generate answer with RAG
    answer = generate_answer_with_rag(translated_text)

    return (f"Detected Language: {detected_lang}\n\n"
            f"Transcription: {text}\n\n"
            f"Transcription (translated to en): {translated_text}\n\n"
            f"RAG Answer: {answer}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    target_lang = request.form.get('target_lang', 'en')
    result = process_audio_query(file, target_lang)

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
