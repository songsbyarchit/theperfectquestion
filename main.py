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
                    "Your task is to write a thoughtful and empathetic acknowledgment that makes the user feel heard and understood. "
                    "The main thing you want to do is use PRAGMATISM or perhaps VERY LIGHT AND SUBTLE POETICISM to SEPERATE the user from what they've said while still making them feel that their concerns/thoughts are VALID"
                    "Almost like a friend explaining VERY LOGICALLY or POETICALLY (not both) explaining why it's OKAY to feel that way, WITHOUT saying 'i also feel that way' or 'we all do' - more seperated and perhaps drawing on evolutionary biology or philosophy or poeticism in a playful way."
                    "Your response must not include ANY imperatives or verbs which TELL the user to DO/FEEL/BE any verb which is prescriptive. Only acknowledgement and nothing else under any cirucmstances."
                    "Your final response must be no longer than 12 words under any circumstances. It MUST be a sentence which flows and has a medium reading level - nothing which is too hard to process or too many obscure words. Avoid passive voice and don't use participle phrases. Make it somewhat informal and playful if appropriate."
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