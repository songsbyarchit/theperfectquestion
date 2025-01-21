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

    1. Reframing Questions: Create vivid, specific scenarios that challenge the user’s assumptions using contrarian reframes tied to their past experiences. At least two of the three questions MUST reference past moments where the opposite of their worry was proven true. Use varied phrasing—not every question needs to begin with "What if." However, include **exactly one** "What if" question in each block. Ensure scenarios are tangible, emotionally engaging, and highlight realistic possibilities that oppose the user’s current perspective.

    2. Force Thinking Questions: Present vivid, high-stakes scenarios that demand immediate, specific responses tailored to the user’s input. Avoid vague or generic phrases like "bold action" or "step forward." Instead, focus on clear, tangible actions or decisions the user could realistically consider. Use time constraints, surprising constraints, or vivid hypothetical situations to provoke urgency and creativity. Ensure questions are direct, actionable, and specific enough to challenge overthinking while sparking decisive ideas.

    3. Belief Questions: Confidently examine the foundations of the user’s beliefs by suggesting EXACTLY TWO specific influences. These must be vivid, realistic, and easy to imagine, avoiding vague references like "societal pressures" or "childhood experiences." Each question must confidently tie the belief to two concrete contexts (e.g., critical feedback at work and family values) and imply they still shape the user’s perspective. Be direct, not tentative—ensure the questions are actionable and grounded in relatable experiences.

    Your questions must extrapolate from the user's input, ensuring they feel deeply connected to their lived experience. Avoid general or abstract phrasing; instead, create engaging and reflective prompts that feel human and supportive.

    Below are examples demonstrating how questions should be generated based on user input:

    **Reframing Questions:**

    **User Input:** "I’m worried about a big presentation tomorrow."  
    1) What if you remembered the last time you spoke confidently and others praised your clarity—how would that change things?  
    2) Think back to when your preparation led to surprising success—what could you replicate this time?  
    3) Could forgetting a detail make you more relatable to the audience, instead of harming your credibility?

    **User Input:** "I feel stuck in my career."  
    1) What if you remembered a past job where feeling stuck led to your next big leap—what might this lead to?  
    2) Recall a time when mastering one new skill opened unexpected doors—what skill might do that now?  
    3) Imagine someone asking for career advice because they admire your growth—what would you share with them?

    **User Input:** "I feel like I’ll never be good at relationships."  
    1) What if someone told you they admired how deeply you listen, even when you feel awkward?  
    2) Recall a time you formed an unexpected connection—what worked well in that moment?  
    3) Imagine a small, kind action this week made someone see you as trustworthy—what would that action be?

    **Force Thinking Questions:**

    User Input: "I’m worried about a big presentation tomorrow."  
    1) If you had just 10 minutes to prepare, what key points would you prioritise?  
    2) If you had to impress someone important with one sentence, what would it be?  
    3) If an audience member had advice for you mid-presentation, what do you imagine they’d say?

    User Input: "I feel stuck in my career."  
    1) If you had one week to make a bold career move, what would it be?  
    2) What’s one skill you could learn right now to feel unstuck?  
    3) If quitting your job meant pursuing your dream, what dream would you chase?

    User Input: "I feel like I’ll never improve at sports."  
    1) If you could only practice for 10 minutes a day, what would you focus on?  
    2) If one tweak in technique made a visible difference, what would you adjust first?  
    3) What would change if a coach guaranteed your improvement after consistent effort for one month?

    **Belief Questions:**

    User Input: "I feel like I’m not good at public speaking."  
    1) Does this stem from school presentations or peer feedback that still shapes your confidence?  
    2) Did a teacher’s criticism or a failed speech plant this belief—how relevant is that now?  
    3) Do memories of school debates or work presentations still influence how you view public speaking?

    User Input: "I’m not creative enough to pursue art."  
    1) Did family expectations or struggles in art class lead to this belief—how valid is it today?  
    2) Did a parent’s focus on practicality or a teacher’s critique limit your confidence in creativity?  
    3) Do family values or past competitions still influence how you see your creative potential?

    User Input: "I feel like I’m not successful enough."  
    1) Does this belief come from parental expectations or peer comparisons still affecting you today?  
    2) Did a sibling’s achievements or workplace pressures shape this—how can you redefine success for yourself?  
    3) Do family benchmarks or academic pressures still shape your idea of what success should look like?

    Now, use the user's input and conversation summary to generate three **Reframing**, three **Force Thinking**, and three **Belief** questions that are deeply specific, engaging, and reflective. Avoid rephrasing what they’ve written; instead, extrapolate to create questions that challenge, inspire, or guide them toward new perspectives.

    User Input: {last_input}

    Conversation Summary: {conversation_summary}

    Return your questions in this format:

    **Reframing Questions:**  
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

    - Ensure exactly one "What if" question per block for variety.  
    - Each question must be concise and contain no more than 20 words under any circumstances.  
    - Reading level MUST be that of a well-read 18 year old and NO HIGHER THAN THIS.  
    - Questions must maintain clarity and provoke thought while remaining direct and engaging.
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