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
            {"role": "system", "content": "You are a playful and cheeky assistant that writes teasing acknowledgments for user journaling."},
            {"role": "user", "content": (
                f"The user wrote:\n{last_input}\n\n"
                "Your task is to write a playful acknowledgment that lightly extrapolates or makes a cheeky assumption based on the user's input. "
                "The goal is to add a bit of humor or charm, perhaps even to the point of slight rudeness/edginess. Keep the reading level MODERATELY LOW and simple sentence structure."
                "For example:\n"
                "1. If the user wrote, 'I'm worried about dating,' respond with, 'Ahh, romantically entangled are we? Happens to the best of us.'\n"
                "2. If the user wrote, 'I missed my morning workout,' respond with, 'Ah, the gym will forgive you this time.'\n"
                "3. If the user wrote, 'I forgot to water my plants,' respond with, 'Oops, plants are patient. They’ll wait for you.'\n"
                "4. If the user wrote, 'I feel stuck in my career,' respond with, 'A little pause is just part of the climb.'\n"
                "5. If the user wrote, 'I can't decide what to eat,' respond with, 'Ah, the eternal food debate!'\n\n"
                "Make your response playful and add a light assumption that flows naturally from the user's input.\n"
                "Your response must NOT include any imperatives, suggestions, presriptions or instructions for the user, NOT even playful ones.\n"
                "Here are three examples of what NOT to do - o verbs should be directed at the user under any circumstances."
                "EXAMPLE 1 of what NOT to do - You should focus on being more confident in your presentation.\n"
                "EXAMPLE 2 of what NOT to do - Try to see this as an opportunity instead of a challenge.\n"
                "EXAMPLE 3 of what NOT to do - Make sure to relax and take deep breaths before your speech.\n\n\n"
                "Your response must be no longer than 10 words under any circumstances and should feel friendly, simple, and easy to read. It must be one sentence maximum."
            )}
        ],
        max_tokens=100,
        temperature=0.8
    )
    acknowledgment = "\n" + acknowledgment_response.choices[0].message['content'].strip()
    final_output = f"{acknowledgment}\n\nMaybe think about this...\n\n{best_question}"

    # Return JSON to the front end
    return jsonify({
        "stage": stage,             # So front-end can update the stage
        "final_output": final_output
    })

if __name__ == "__main__":
    app.run(port=8000)  # Replace 8000 with your desired port number