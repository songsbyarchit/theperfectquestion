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
    """
    Asks OpenAI for 3 out-of-the-box questions relevant to the stage.
    Each question should have 3 talking points. And no more then 3 under any circumstances. (sub questions or prompts).
    """

    prompt = f"""
    The user is currently in the '{stage}' stage of reflection. Your task is to generate **3 distinct questions** tailored to this stage. Each question must include **3 talking points. And no more then 3 under any circumstances.** with suggestions on how the user can expand their reflection. Avoid encouraging repetitive or circular journaling.

    Descriptions of stages:
    - **Description**: Focus on recounting events or experiences. Questions should help the user clarify details and organize their thoughts.
    - **Processing**: Focus on identifying and unpacking emotions. Questions should encourage naming feelings and understanding emotional responses.
    - **Analysis**: Focus on exploring underlying causes. Questions should guide the user toward deeper insights about their thoughts, patterns, or behaviors.
    - **Planning**: Focus on actionable steps or forward-thinking. Questions should help the user set intentions, define goals, or plan for the future.

    Examples for each stage:
    - **Description**:
    1) What happened during this experience?  
        - Write down key events in chronological order.
        - Focus on specific moments that stood out to you.
        - Avoid analyzing or judging the events—just describe them.
    
    - **Processing**:
    1) What emotions are you experiencing right now?  
        - Name the feelings you're going through (e.g., anger, sadness, joy).  
        - Reflect on when these emotions began and how they’ve evolved.  
        - Consider whether these emotions are connected to specific events or thoughts.

    - **Analysis**:
    1) Why do you think this happened?  
        - Identify possible triggers or causes for this event.  
        - Reflect on patterns that might be repeating in your life.  
        - Explore whether any assumptions or beliefs influenced your actions.

    - **Planning**:
    1) What steps can you take to address this situation moving forward?  
        - Identify one small, actionable step you can take immediately.  
        - Think about larger principles you want to apply in similar situations.  
        - Set a goal or intention for the next time you face this type of challenge.

    Now, generate 3 questions tailored to the '{stage}' stage, each with 3 talking points and no more under any circumstances
    
    Each bullet point should be somewhat abstract but not shocking. i.e. it should't be completely seperate from the MAIN QUESTION.
    
    But it SHOULD add personality and force the user to critically reflect in 3 different directions starting from the main question, to enable "bigger thinking" and "critical thinking."

    Return the output in this format:

    1) Main question goes here
    - talking point 1
    - talking point 2
    - talking point 3

    2) Main question goes here
    - talking point 1
    - talking point 2
    - talking point 3

    3) Main question goes here
    - talking point 1
    - talking point 2
    - talking point 3
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates 3 out-of-the-box questions for reflection stages."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message['content'].strip()

# This function picks the SINGLE best question out of the 3 by calling OpenAI again
def pick_best_question(five_questions_text, last_input, conversation_summary, stage):
    """
    Takes the 3 questions generated, and calls OpenAI to pick the best one
    given the user's current stage, last_input, and conversation_summary.
    """
    prompt = f"""
    We have these 3 questions generated for the user:

    {five_questions_text}

    The user's most recent input:
    {last_input}

    Conversation summary so far:
    {conversation_summary}

    Your task is to pick the **single best question** from the list above. Consider the following criteria:
    1) The question must align with the user's current stage of reflection ('{stage}').
    2) The question must avoid repetition of what the user has already explored.
    3) The question should encourage the user to move forward in their reflection, offering new insights or actionable ideas.

    Examples of "best" questions:
    - For **Description**: "What specific events or details from today stand out the most, and why?"
    - For **Processing**: "What is one emotion you're feeling right now, and what might have triggered it?"
    - For **Analysis**: "What patterns do you notice in situations like this, and how do they affect you?"
    - For **Planning**: "What is one step you can take to address this challenge, and how will you ensure you stick to it?"

    Respond ONLY with the single best question and its talking points. And no more then 3 under any circumstances., formatted as:

    Main question goes here (ending in a question mark - no hyphem before it)
    - Bullet point 1 (not ending in a question mark, phrased open-mindedly and not like an exam question)
    - Bullet point 2 (not ending in a question mark, phrased open-mindedly and not like an exam question)
    - Bullet point 3 (not ending in a question mark, phrased open-mindedly and not like an exam question)
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that selects the single best reflection question based on the user's input and stage."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    best_question = response.choices[0].message['content'].strip()
    return best_question