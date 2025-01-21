import openai
import logging
logging.basicConfig(level=logging.INFO)

# This function uses OpenAI to figure out which stage user is in
def detect_stage(last_input, conversation_summary):
    """
    Calls OpenAI to classify which of the 4 stages (description, processing,
    analysis, planning) the user is currently demonstrating.
    """

    prompt = f"""
    The user last wrote:
    {last_input}

    Conversation summary so far:
    {conversation_summary}

    Your task is to determine which stage of journaling reflection the user is in. Use the following detailed descriptions of stages:

    1) **Description**: The user is recounting events or experiences in a factual way without mentioning emotions or deeper analysis. This stage focuses on what happened, providing details like who, what, where, and when. There is no emotional unpacking or reasoning involved. Keywords: "It was," "I went," "It happened."
    2) **Processing**: The user is identifying or unpacking emotions. They are reflecting on their emotional state, naming specific feelings, or trying to understand what they are going through emotionally. Keywords: "I feel," "I am experiencing," "I noticed."
    3) **Analysis**: The user is exploring underlying causes or reasons. They are analyzing why certain events occurred or why they feel a certain way. They may seek patterns, insights, or deeper reasoning. Keywords: "Why did this happen," "I think it is because," "I realize."
    4) **Planning**: The user is focused on actionable steps or thinking about the future. They are creating goals, setting intentions, or thinking about principles to apply going forward. Keywords: "I will," "Next time," "Steps to take."

    Examples for reference:
    - Example of **Description**: "I had a stressful day at work today. I had two meetings and missed a deadline."
    - Example of **Processing**: "I feel frustrated and anxious because I couldn’t meet the deadline. I’m also worried about how others perceive me."
    - Example of **Analysis**: "I think my frustration comes from setting unrealistic expectations for myself and being a perfectionist."
    - Example of **Planning**: "Next time, I’ll break tasks into smaller steps and set more achievable deadlines."

    Based on the user's writing and conversation summary, respond ONLY with the stage name: 'description', 'processing', 'analysis', or 'planning'.

    If multiple stages seem to be mentioned, pick the one which is the BEST fit and prioritize the user's last input. If the input is purely factual and lacks emotional or deeper reasoning, always choose 'description'.
    """
    
    # Example OpenAI call (GPT-3.5, etc.)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that determines the user's journaling reflection stage."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.0
    )
    
    # Extract the stage name (you could refine parsing as needed)
    stage = response.choices[0].message['content'].strip().lower()
    # Safety check, default to 'description' if not recognized
    if stage not in ["description", "processing", "analysis", "planning"]:
        stage = "description"

    logging.info(f"Detected Stage: {stage}")
    return stage

# This function generates 3 distinct questions for the determined stage.
def generate_questions_for_stage(stage, last_input, conversation_summary):
    prompt = f"""
    The user is currently in the '{stage}' stage of reflection. Your task is to generate exactly **three questions** for EACH of the following categories:

    1. "What if" questions: These are hypothetical, imaginative questions designed to challenge assumptions and expand the user’s perspective. They should reframe the situation by introducing alternative possibilities that feel vivid and tangible. The goal is to inspire curiosity and help the user see their thoughts or challenges from an entirely new angle.

    2. Force thinking questions: These are sharp, direct questions meant to generate quick, actionable ideas without overthinking. They should push the user out of their usual thought patterns, creating a sense of urgency or constraint to provoke fresh, practical insights. Each question should demand a response that cuts through indecision or complexity.

    3. Belief questions: These are deep, probing questions that focus on uncovering the foundations of the user’s thoughts and beliefs. They should drill into the core of why the user feels or thinks a certain way, exploring origins, contradictions, or evidence. The aim is to invite introspection and challenge the user to critically examine the strength and validity of their beliefs.

    Each question must be distinct, engaging, and appropriate for the '{stage}' stage. Return your questions in this format:

    **What if Questions:**
    1) QUESTION
    2) QUESTION
    3) QUESTION

    **Force Thinking Questions:**
    1) QUESTION
    2) QUESTION
    3) QUESTION

    **Belief Questions:**
    1) QUESTION
    2) QUESTION
    3) QUESTION
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates diverse reflection questions for user journaling."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

# This function picks the best question from each category ("What if," "Force Thinking," "Belief") by calling OpenAI
def pick_best_question(questions_text, last_input, conversation_summary, stage):
    """
    Takes the questions generated and selects one best question from each category
    ("What if," "Force Thinking," "Belief") based on the user's current stage, last_input, and conversation_summary.
    """
    prompt = f"""
    Here are the questions generated for the user:

    {questions_text}

    The user's most recent input:
    {last_input}

    Conversation summary so far:
    {conversation_summary}

    Your task is to select exactly one question from EACH of the following categories based on the existing questions provided:
    - "What if" questions: Hypothetical questions that challenge assumptions, expand possibilities, and help reframe perspectives.
    - Force Thinking questions: Direct, sharp questions that provoke actionable ideas and encourage quick thinking without overanalyzing.
    - Belief questions: Probing questions that explore the roots of the user's beliefs, examining why they think or feel a certain way.

    Selection criteria:
    1) The question must align with the user's current stage of reflection ('{stage}').
    2) The question must be thought-provoking, offering the user new insights or ways to approach their reflection.
    3) The question must avoid repetition and feel distinct, ensuring diversity in the types of reflections it inspires.
    4) Select only from the provided questions. Do NOT invent or generate new questions.

    Respond with exactly three questions in this order, without any titles, headings, or additional commentary:

    [What if question]

    [Force Thinking question]

    [Belief question]
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that selects the best reflection questions from provided options."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    best_questions = response.choices[0].message['content'].strip()
    return best_questions