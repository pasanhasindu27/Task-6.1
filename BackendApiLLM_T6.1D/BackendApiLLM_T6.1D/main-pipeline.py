import os
import requests
from flask import Flask, request, jsonify
import re

app = Flask(__name__)

# API setup
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
print("Hugging Face API Token:", HF_API_TOKEN)  # Debugging line
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def fetchQuizFromLlama(student_topic):
    print("Fetching quiz from Hugging Face Inference API")

    prompt = (
        f"Generate a quiz with exactly 3 questions about: {student_topic}\n"
        f"For each question:\n"
        f"- Phrase the question clearly\n"
        f"- Provide 4 distinct options labeled A-D\n"
        f"- Indicate the correct answer with just the letter\n"
        f"Use this exact format:\n\n"
        f"**QUESTION 1:** [Question text]?\n"
        f"**OPTION A:** [Choice 1]\n"
        f"**OPTION B:** [Choice 2]\n"
        f"**OPTION C:** [Choice 3]\n"
        f"**OPTION D:** [Choice 4]\n"
        f"**ANS:** A\n\n"
        f"Repeat for Questions 2 and 3 with the same format."
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,  
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if not isinstance(result, list) or 'generated_text' not in result[0]:
            raise ValueError("Unexpected API response format")
            
        generated_text = result[0]['generated_text']
        quiz_start = generated_text.find("**QUESTION 1:**")
        
        if quiz_start == -1:
            raise ValueError("Quiz format not detected in response")
            
        return generated_text[quiz_start:]
        
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")

def process_quiz(quiz_text):
    questions = []
    # More robust regex pattern with optional whitespace and markdown
    pattern = re.compile(
        r'(?i)\*\*QUESTION\s*\d+:\*\*\s*(.+?)\s*'  # Case-insensitive, flexible spacing
        r'\*\*OPTION\s*A:\*\*\s*(.+?)\s*'
        r'\*\*OPTION\s*B:\*\*\s*(.+?)\s*'
        r'\*\*OPTION\s*C:\*\*\s*(.+?)\s*'  # Tolerate typo in OPTION
        r'\*\*OPTION\s*D:\*\*\s*(.+?)\s*'
        r'\*\*ANS:\*\*\s*([A-D])',  # Strict letter matching
        re.DOTALL
    )

    matches = pattern.findall(quiz_text)
    
    for match in matches:
        try:
            question = match[0].strip().rstrip('?') + '?'  # Ensure question ends with ?
            options = {
                'A': match[1].strip(),
                'B': match[2].strip(),
                'C': match[3].strip(),
                'D': match[4].strip()
            }
            correct = match[5].upper().strip()  # Normalize to uppercase
            
            if correct not in options:
                continue  # Skip invalid answers
                
            questions.append({
                "question": question,
                "options": options,
                "correct_answer": correct
            })
        except (IndexError, TypeError):
            continue  # Skip malformed questions
            
    return questions[:3]  # Return max 3 questions

@app.route('/getQuiz', methods=['GET'])
def get_quiz():
    student_topic = request.args.get('topic')
    if not student_topic or len(student_topic.strip()) < 3:
        return jsonify({'error': 'Valid topic parameter required (min 3 characters)'}), 400
        
    try:
        quiz_text = fetchQuizFromLlama(student_topic.strip())
        processed_quiz = process_quiz(quiz_text)
        
        if not processed_quiz:
            return jsonify({
                'error': 'Failed to generate valid quiz questions',
                'debug': quiz_text[:500]  # Limited debug info
            }), 500
            
        return jsonify({
            'topic': student_topic,
            'count': len(processed_quiz),
            'questions': processed_quiz
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Quiz generation failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")