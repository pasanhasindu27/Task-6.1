import os
import requests
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

# Hugging Face API setup
HF_API_TOKEN = os.getenv('HF_API_TOKEN', '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')
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN environment variable not set")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def fetchQuizFromHuggingFace(student_topic):
    print("Fetching quiz from Hugging Face Inference API")
    
    payload = {
        "inputs": (
            f"Generate a quiz with 8 questions to test students on the topic '{student_topic}'. "
            f"For each question, generate 4 options where only one is correct. "
            f"Strict format:\n"
            f"**QUESTION 1:** [Question]\n"
            f"**OPTION A:** [Option]\n"
            f"**OPTION B:** [Option]\n"
            f"**OPTION C:** [Option]\n"
            f"**OPTION D:** [Option]\n"
            f"**ANS:** [Correct letter]\n\n"
            f"**QUESTION 2:** [Question] ...\n"
            f"**QUESTION 3:** [Question] ...\n"
        ),
        "parameters": {
            "max_new_tokens": 700,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        output = response.json()
        # output is usually a list of generated text
        if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
            return output[0]['generated_text']
        else:
            return output
    else:
        raise Exception(f"HF API request failed: {response.status_code} - {response.text}")

def process_quiz(quiz_text):
    questions = []
    try:
        # Match the exact format with ** markers
        pattern = re.compile(
            r'\*\*QUESTION \d+:\*\*\s*(.+?)\s*'
            r'\*\*OPTION A:\*\*\s*(.+?)\s*'
            r'\*\*OPTION B:\*\*\s*(.+?)\s*'
            r'\*\*OPTION C:\*\*\s*(.+?)\s*'
            r'\*\*OPTION D:\*\*\s*(.+?)\s*'
            r'\*\*ANS:\*\*\s*([A-D])',
            re.DOTALL | re.IGNORECASE
        )

        matches = pattern.findall(quiz_text)
        
        for match in matches:
            questions.append({
                "question": match[0].strip(),
                "options": [
                    match[1].strip(),
                    match[2].strip(),
                    match[3].strip(),
                    match[4].strip()
                ],
                "correct_answer": match[5].upper()
            })
            
        return questions[:4]  # Return max 3 questions
    
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return get_fallback_quiz()

def get_fallback_quiz():
    return [{
        "question": "Who was the first US President?",
        "options": ["George Washington", "Thomas Jefferson", 
                    "Abraham Lincoln", "John Adams"],
        "correct_answer": "A"
    }]

@app.route('/getQuiz', methods=['GET'])
def get_quiz():
    print("Request received")
    student_topic = request.args.get('topic')
    if not student_topic:
        return jsonify({'error': 'Missing topic parameter'}), 400
    try:
        quiz = fetchQuizFromHuggingFace(student_topic)
        print(quiz)
        processed_quiz = process_quiz(quiz)
        if not processed_quiz:
            return jsonify({'error': 'Failed to parse quiz data', 'raw_response': quiz}), 500
        return jsonify({'quiz': processed_quiz}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def run_test():
    return jsonify({'quiz': "test"}), 200

if __name__ == '__main__':
    port_num = 5000
    print(f"App running on port {port_num}")
    app.run(port=port_num, host="0.0.0.0")
