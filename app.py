import os
import pyaudio
import wave
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from google.cloud import speech
import google.generativeai as genai
from threading import Thread
from flask import Flask, send_from_directory

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
genai.configure(api_key=os.getenv("Gemini_API_Key"))


app = Flask(__name__, static_folder="build")
CORS(app) 

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

# Global variables to control recording state
is_recording = False
frames = []

prompts = {
    "part_1": [
        "Can you tell me about yourself?",
        "What do you do? Do you work or study?",
        "What are your hobbies?",
        "Do you like traveling? Why or why not?"
    ],
    "part_2": [
        "Describe a memorable trip you took. You should say:\n- Where you went\n- Who you went with\n- What you did\nAnd explain why it was memorable.",
        "Describe a favorite book you’ve read. You should say:\n- What it is\n- Who wrote it\n- What it is about\nAnd explain why you enjoyed it."
    ],
    "part_3": [
        "Why do you think people enjoy traveling to new places?",
        "How does reading books contribute to personal growth?",
        "What is the importance of leisure activities in a busy lifestyle?"
    ]
}

# Track test progress (simplified; can be enhanced with a database)
test_progress = {}

@app.route('/start_test', methods=['POST'])
def start_test():
    user_id = request.json.get('user_id', 'default_user')
    test_progress[user_id] = {"current_section": "part_1", "question_index": 0}
    question = prompts["part_1"][0]
    return jsonify({"section": "part_1", "question": question})

@app.route('/next_question', methods=['POST'])
def next_question():
    user_id = request.json.get('user_id', 'default_user')
    progress = test_progress.get(user_id, {})
    section = progress.get("current_section", "part_1")
    question_index = progress.get("question_index", 0)

    if section == "part_1":
        if question_index < len(prompts["part_1"]) - 1:
            question_index += 1
            test_progress[user_id]["question_index"] = question_index
        else:
            section = "part_2"
            question_index = 0
            test_progress[user_id]["current_section"] = section
            test_progress[user_id]["question_index"] = question_index

    elif section == "part_2":
        if question_index < len(prompts["part_2"]) - 1:
            question_index += 1
            test_progress[user_id]["question_index"] = question_index
        else:
            section = "part_3"
            question_index = 0
            test_progress[user_id]["current_section"] = section
            test_progress[user_id]["question_index"] = question_index

    elif section == "part_3":
        if question_index < len(prompts["part_3"]) - 1:
            question_index += 1
            test_progress[user_id]["question_index"] = question_index
        else:
            return jsonify({"message": "Test completed!"})

    question = prompts[section][question_index]
    return jsonify({"section": section, "question": question})

@app.route('/api/start-record', methods=['POST'])
def start_record():
    global is_recording, frames
    is_recording = True
    frames = []

    # Start recording in a separate thread
    def record():
        global is_recording, frames
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        while is_recording:
            data = stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    thread = Thread(target=record)
    thread.start()

    return jsonify({"message": "Recording started"})

@app.route('/api/stop-record', methods=['POST'])
def stop_record():
    global is_recording, frames
    is_recording = False

    # Save audio to a .wav file
    filename = "audio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    return jsonify({"message": "Recording stopped", "filename": filename})

def transcribe_audio(filename):
    client = speech.SpeechClient()
    
    with open(filename, 'rb') as audio_file:
        content = audio_file.read()

    # The name of the audio file to transcribe
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Print the transcription result
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))
        return result.alternatives[0].transcript
    return ""
def get_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"You are an IELTS examiner. Respond to the following candidate answer: '{user_input}'"
        )
        print(f"User Input: {user_input}")  # Log user input
        print(f"AI Response: {response.text}")  # Log AI response
        return response.text
    except Exception as e:
        return str(e)
    
@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.get_json()
    user_transcription = data.get('transcription', '')

    if not user_transcription:
        return jsonify({"error": "No transcription provided"}), 400

    try:
        # Generate response using Gemini
        ai_response = get_response(user_transcription)
        return jsonify({"ai_response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

@app.route("/api/record", methods=["POST"])
def record():
    audio_file = start_record()
    return jsonify({"status": "success", "audio_file": audio_file})


# Flask route to handle transcription
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    transcription = transcribe_audio(filename)
    print(f"Transcription: {transcription}")  # Log transcription
    return jsonify({"transcription": transcription})

# Flask route to generate response
@app.route("/api/respond", methods=["POST"])
def respond():
    data = request.get_json()
    user_input = data.get("user_input")
    if not user_input:
        return jsonify({"error": "User input is required"}), 400

    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)