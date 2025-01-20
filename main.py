from flask import Flask, request, jsonify, render_template
import openai
from questions import detect_stage, generate_questions_for_stage, pick_best_question
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()  # Load environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reflect", methods=["POST"])
def reflect():
    data = request.get_json()
    last_input = data.get("last_input", "")
    conversation_summary = data.get("conversation_summary", "")
    current_stage = data.get("current_stage", "")

    # 1) Detect the stage if none provided or always re-detect
    stage = detect_stage(last_input, conversation_summary)

    # 2) Generate 5 questions for that stage
    five_questions_text = generate_questions_for_stage(stage, last_input, conversation_summary)

    # 3) Use a second call to pick the single best question
    best_question = pick_best_question(five_questions_text, last_input, conversation_summary, stage)

    # 4) Build an acknowledgment + the final question
    #    You can customize how you want to “acknowledge” user input
    acknowledgment_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and empathetic assistant that writes thoughtful acknowledgments for user journaling."},
            {"role": "user", "content": (
                f"The user wrote:\n{last_input}\n\n"
                "Based on their input, write a thoughtful, warm empathetic acknowledgment that encourages them to reflect further. "
                "Make it feel human-like, engaging, and supportive. If possible, highlight one specific part of their input to show understanding, "
                "and gently guide them to keep journaling. It must be no longer than 12 words under any circumstances"
            )}
        ],
        max_tokens=100,
        temperature=0.7
    )
    acknowledgment = acknowledgment_response.choices[0].message['content'].strip()
    final_output = f"{acknowledgment}\n\nHere’s a question you might explore:\n{best_question}"

    # Return JSON to the front end
    return jsonify({
        "stage": stage,             # So front-end can update the stage
        "final_output": final_output
    })

if __name__ == "__main__":
    app.run(port=8000)  # Replace 8000 with your desired port number