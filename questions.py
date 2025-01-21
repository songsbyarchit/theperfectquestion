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

    1. "What if" questions: Hypothetical, imaginative questions designed to challenge assumptions and expand the user’s perspective. They should introduce vivid alternative possibilities and help the user imagine entirely new angles to their situation.

    2. Force Thinking questions: Sharp, direct questions designed to provoke quick, actionable ideas without overthinking. They should push the user out of their usual patterns, forcing them to act or think decisively.

    3. Belief questions: Probing, introspective questions that examine the foundations of the user’s thoughts and beliefs. These questions must explicitly suggest two realistic and plausible sources where the belief may have originated based on the user's input. Examples of sources could include:
   - Personal experiences in school, work, or hobbies
   - Influences from family dynamics or societal expectations
   - Interactions with peers, mentors, or past authority figures

    Your questions must extrapolate from the user's input, ensuring they feel deeply connected to their lived experience. Avoid general or abstract phrasing; instead, create engaging and reflective prompts that feel human and supportive.

    Your questions must be extremely specific to the user's last input and the provided conversation summary. Avoid generic rephrasing. Instead, extrapolate and build on the user's input to make them think bigger, challenge assumptions, and explore deeper reflections. Use your responses to add human-ness to the journaling process, making it feel engaging, personal, and surprising.

    Below are examples demonstrating how questions should be generated based on user input:

    **Examples of User Input and Generated Questions**:

    User Input: "I feel stuck in my career."
    **What if Questions:**
    1) What if you could instantly switch to a different career—what would it be?
    2) What if being “stuck” is actually an opportunity to slow down and reflect—how would that feel?
    3) What if someone in your life saw your situation as inspiring—what might they admire?

    **Force Thinking Questions:**
    1) If you had one week to make a bold career move, what would it be?
    2) What’s one skill you could learn right now to feel unstuck?
    3) What would you do today if failure wasn’t an option?

    Example structure for Belief Questions:
    1) Why do you believe [insert belief]? Could this stem from [Suggestion 1] or [Suggestion 2]?
    2) What experiences in [Suggestion 1] or [Suggestion 2] might have shaped this belief?
    3) Do you think [Suggestion 1] or [Suggestion 2] still influence your perspective today?

    User Input: "I’m worried about a big presentation tomorrow."
    **What if Questions:**
    1) What if you gave the presentation to your best friend instead—how would it feel different?
    2) What if the audience is rooting for you more than you think—how does that change your confidence?
    3) What if this presentation could open doors to something you’ve always wanted—what’s the best-case outcome?

    **Force Thinking Questions:**
    1) If you had 10 minutes to prepare, what key points would you focus on?
    2) What’s one unexpected question the audience might ask, and how would you answer?
    3) What’s one thing you can do right now to calm your nerves?

    **Belief Questions:**
    1) Why do you believe this presentation defines your abilities—what else defines you?
    2) Is there a past experience influencing your fear about tomorrow?
    3) What evidence do you have that this will go badly—what’s missing in that picture?

    Now, use the user's input and conversation summary to generate three **What if**, three **Force Thinking**, and three **Belief** questions that are deeply specific, engaging, and reflective. Avoid rephrasing what they’ve written; instead, extrapolate to create questions that challenge, inspire, or guide them toward new perspectives.

    User Input: {last_input}

    Conversation Summary: {conversation_summary}

    Return your questions in this format:

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
            {"role": "system", "content": "You are a helpful assistant that generates specific, creative, and thought-provoking questions tailored to user journaling."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.8
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