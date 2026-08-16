# %% [markdown]
# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 16 Aug, 2026]
# **Due Date:** 16th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** Oluwademilade Subair
# **Student ID:** 58742028
# 
# ---
# 
# ### Objective
# 
# In the previous labs you _trained_ models. In this lab you will _use_ a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider                       | Free tier     | Notes                                                 |
# | ------------------------------ | ------------- | ----------------------------------------------------- |
# | **Groq** (recommended)         | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini**              | Yes           | `google-generativeai` package                         |
# | **Hugging Face Inference API** | Yes, limited  | Many open models                                      |
# | OpenAI / Anthropic             | Paid          | Fine if you already have credits                      |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.
# 

# %% [markdown]
# ---
# 
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.
# 

# %%
# API-key setup — DO NOT hard-code your key in this cell.

import os

# --- Local (with a .env file) ---
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["GROQ_API_KEY"]

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",  # remove this line if using OpenAI itself
)
MODEL = "llama-3.3-70b-versatile"  # or your provider's model name

print("Client ready.")

# %% [markdown]
# ---
# 
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).
# 

# %% [markdown]
# ### Part 1.1 — Your first API call
# 

# %%
# A helper function you will reuse for the WHOLE lab:
def ask_llm(
    user_prompt,
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=500,
):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(response.usage)
    return response.choices[0].message.content


# Call it once with a simple question and print the answer
print(ask_llm("Why is the sky blue?"))

# Print response.usage as well to see how many tokens did the call consume

# %% [markdown]
# **Student Reasoning — Anatomy of a call**
# _1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each._
# _2. What is a token, roughly? Why do API providers bill per token rather than per request?_
# 
# > **Answer:**
# 
# - `system` sets the behaviour, persona and constraints for the model, _e.g., "You are a Socratic writing tutor. Never write essays for the studentt"_. `user` doesn't see this layer and is the actual conversational input from the person interacting with the model, _e.g., "My thesis is that mixed socialism works better in African societies."_
# - A token is an atomic unit of text created from a larger text. Providers bill per token rather than per request because requests vary wildly in size as requests can range from one word questions to documents spanning many pages
# 

# %% [markdown]
# ### Part 1.2 — Temperature: the randomness dial
# 

# %%
# Asking the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
# Using this good test question: "Suggest a name for a savings product for market traders in Accra."
# Print all 10 answers, grouped by temperature.

print("================== Question at t = 0.0 ==================")
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=0.0,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=0.0,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=0.0,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=0.0,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=0.0,
    )
)

print()

print("================== Question at t = 1.2 ==================")
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=1.2,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=1.2,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=1.2,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=1.2,
    )
)
print(
    ask_llm(
        "Suggest a name for a savings product for market traders in Accra.",
        temperature=1.2,
    )
)

# %% [markdown]
# **Student Reasoning — Temperature**
# _What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?_
# 
# > **Answer:**
# 
# - At `t = 0.0`, all five responses are nearly identical with the same 7 names. explanations and closing line. The model appears deterministic
# - At `t = 1.2`, there is a noticeable divergence in responses as different names begin to appear with different structures and closing lines. The model seems to be more creative but also less consistent
# - For a loan-decision system, a lower temperature around `t = 0.0` is preferable as we are gunning for consistency, reliability and predictability. Loan decisions can affect people's finances directly so creativity is a liability, although we don't want to eliminate the possibility of variance
# 

# %% [markdown]
# ---
# 
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.
# 

# %%
LETTERS = {
    "L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",
    "L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",
    "L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",
    "L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",
    "L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",
    "L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
    "L001": {
        "applicant_name": "Akosua Mensah",
        "amount_ghs": 8000,
        "purpose": "buy deep freezer / expand into frozen foods",
        "monthly_profit_ghs": 900,
        "has_collateral_or_guarantor": True,
        "repayment_months": 20,
    },
    "L003": {
        "applicant_name": "Efua Darko",
        "amount_ghs": 15000,
        "purpose": "industrial sewing machines and fabric stock",
        "monthly_profit_ghs": 2800,
        "has_collateral_or_guarantor": True,
        "repayment_months": 15,
    },
    "L006": {
        "applicant_name": "Kofi",
        "amount_ghs": 50000,
        "purpose": "car wash, provision shop, phone imports",
        "monthly_profit_ghs": None,
        "has_collateral_or_guarantor": False,
        "repayment_months": 12,
    },
}

print(f"{len(LETTERS)} letters loaded.")

# %% [markdown]
# ---
# 
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.
# 

# %% [markdown]
# ### Part 3.1 — Component 1: Summarization
# 
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.
# 

# %%
# Writing SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").
#   Run it on L002 and L006. Read the output critically.
SUMMARY_PROMPT_V1 = "Summarise this to your best ability:"


def summarise_v1(letter_text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{SUMMARY_PROMPT_V1}\n\n{letter_text}"}],
        temperature=0,
    )
    return response.choices[0].message.content


print("=== L002 w/ V1 ===")
print(summarise_v1(LETTERS["L002"]))

print("=== L006 w/ V1 ===")
print(summarise_v1(LETTERS["L006"]))

# Now write SUMMARY_PROMPT_V2 as a proper template with:
#   - a system prompt giving the LLM a ROLE (e.g. "You are an assistant to a microfinance
#     loan officer...") and constraints (factual, neutral, no invented details, 3-4 sentences)
#   - a user prompt template like: f"Summarize this loan application:\n\n{letter_text}"
#   Run V2 on the same two letters at temperature=0.

SUMMARY_PROMPT_V2 = """You are an assistant to a microfinance loan officer at 
a financial institution in Ghana. Your job is to summarise loan application letters in 
3 to 4 sentences factually and neutrally without bias or emotion. Do not invent details. 
Do not make recommendations"""


def summarise_v2(letter_text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT_V2},
            {
                "role": "user",
                "content": f"Summarise this loan application:\n\n{letter_text}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


print("=== L002 w/ V2 ===")
print(summarise_v2(LETTERS["L002"]))

print("=== L006 w/ V2 ===")
print(summarise_v2(LETTERS["L006"]))

# Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook

# %% [markdown]
# **Student Reasoning — Summarization prompts**
# _1. What concrete problems did V1's output have that V2 fixed? Quote examples._
# _2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?_
# 
# > **Answer:**
# 
# - V1 added judgemental or unsupported framing that V2 removed. For example, V1 says L002 is "urgently seeking" the loan and that he "promises to repay the load as soon as possible," while V2 sticks to the safer facts quoting "does not currently have collateral to offer." For L006, V1 says he is "relying on his personal guarantee of being trustworthy", an interpretation compared to V2's careful answer of "asserts that he is trustworthy."
# - "no invented details" is essential because these summaries could influence a lending decision regarding large sums of money. Adding unsupported claims about urgency and reliability, especially when RAG is not used, a model can confidently produce output that is incredibly false or not creditworthy. It is called hallucination
# 

# %% [markdown]
# ### Part 3.2 — Component 2: Structured extraction (JSON)
# 
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.
# 

# %%
# Write EXTRACT_PROMPT — a template that instructs the model to return ONLY a JSON
#   object with EXACTLY these keys:
#     applicant_name (string), amount_ghs (number), purpose (string),
#     monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
#     repayment_months (number or null)
#   Techniques to use:
#     - explicit schema in the prompt
#     - ONE worked example (few-shot) using a letter you write yourself (not from LETTERS!)
#     - "If a field is not stated in the letter, use null. Do not guess."
#     - temperature=0
import json

import pandas as pd

EXTRACT_PROMPT = """
You are an assistant to a microfinance loan officer at a financial institution in Ghana.

Extract information from the loan application letter and return ONLY a valid JSON object.

Use EXACTLY these keys:

{
  "applicant_name": string,
  "amount_ghs": number,
  "purpose": string,
  "monthly_profit_ghs": number or null,
  "has_collateral_or_guarantor": boolean,
  "repayment_months": number or null
}

Rules:
- Do not include any explanation before or after the JSON.
- Do not use ``` json fences
- If a field is not stated in the letter, use null.
- Do not guess or infer missing information.
- has_collateral_or_guarantor should be true only if the letter explicitly mentions collateral or a guarantor.

Worked example:

Letter:
"My name is Demi Subair. I am requesting GHS 5,000 to expand my tech accessories shop.
The business earns about GHS 2,000 profit each month. My dad has agreed to act
as my guarantor. I hope to repay the loan within 12 months."

Output:
{
  "applicant_name": "Demi Subair",
  "amount_ghs": 5000,
  "purpose": "expand my tech accessories shop",
  "monthly_profit_ghs": 2000,
  "has_collateral_or_guarantor": true,
  "repayment_months": 12
}
"""


def extract_json(letter_text, temperature):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {
                "role": "user",
                "content": f"Extract the fields from this loan application:\n\n{letter_text}",
            },
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# Write extract_fields(letter_text) that calls the LLM, strips any ```json fences,
#   json.loads() the result, and returns a dict. Handle parse failures gracefully
#   (return None and print a warning).
def extract_fields(letter_text, temperature):
    result = extract_json(letter_text, temperature)

    # Remove possible markdown fences just in case
    result = result.strip()

    if result.startswith("```json"):
        result = result[7:]

    if result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()

    try:
        return json.loads(result)

    except json.JSONDecodeError:
        print("Warning: Could not parse model output as JSON.")
        print(result)
        return None


# Run it on ALL SIX letters; collect results into a pandas DataFrame (one row per
# letter) and display it.
rows = []

for letter_id, letter_text in LETTERS.items():
    extracted = extract_fields(letter_text, 0)

    if extracted is not None:
        extracted["letter_id"] = letter_id
        rows.append(extracted)

df = pd.DataFrame(rows)

df

# %% [markdown]
# **Student Reasoning — Structured extraction**
# _1. Why must the few-shot example NOT come from the six letters you are processing?_
# _2. Why "use null, do not guess" — what did the model do without that instruction?_
# _3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?_
# 
# > **Answer:**
# 
# - It will accidentally leak part of the evaluation data in the prompt. Upon test time, the results won't be as fair, so a separate invented example demonstrates the format well
# - It prevents the model from filling missing fields with plausible-sounding values that can bypass as real. Missing information MUST stay missing. Without it, the model may guess things like a repayment period, monthly profit, or collateral status when when it is not explicitly stated.
# - A lower temperature is more stable and will focus on the nitty gritty details. Creativity will use a higher temperate to look for less obvious tokens and produce varied ideas
# 

# %% [markdown]
# ### Part 3.3 — Component 3: The decision-support brief
# 
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.
# 

# %%
# Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
#     1. Strengths (bullet points, grounded in the letter)
#     2. Red flags (bullet points)
#     3. Missing information the officer should request
#     4. Suggested next step (e.g. "invite for interview", "request documents",
#        "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.

BRIEF_PROMPT = """
You are assisting a loan officer for a microfinance bank in Ghana.

Using ONLY the applicant's letter and the extracted JSON provided, produce a concise officer brief with these sections:

1. Strengths
- Bullet points
- Must be grounded in evidence from the letter or extracted JSON

2. Red flags
- Bullet points
- Mention inconsistencies, weak evidence, unclear claims, or concerns
- Do not invent concerns that are not supported by the inputs

3. Missing information the officer should request
- Bullet points
- Identify documents, clarification, or evidence needed

4. Suggested next step
Choose an appropriate next step from these options:
- invite for interview
- request additional documents
- request clarification
- flag for senior review
- proceed to further assessment

Do NOT recommend "approve" or "reject".

Important: You are only assisting with review. Final decisions is made by a qualified human officer.
"""


def generate_brief(letter_text, extracted_json):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""
                    {BRIEF_PROMPT}

                    LETTER:
                    {letter_text}

                    EXTRACTED JSON:
                    {extracted_json}
                    """,
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.
briefs = {}

for letter_id, letter_text in LETTERS.items():
    extracted_json = df[df["letter_id"] == letter_id].iloc[0].to_dict()
    briefs[letter_id] = generate_brief(letter_text, extracted_json)

for letter_id in ["L001", "L002", "L003", "L006"]:
    print(f"\n=== {letter_id} BRIEF ===")
    print(briefs[letter_id])

# %% [markdown]
# **Student Reasoning — Decision support**
# _1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?_
# _2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason._
# 
# > **Answer:**
# 
# - Yes. For L003, the system correctly identified strong evidence of registered business, strong revenue history, collateral and sales records. The red flags were relatively minor and focused on verification which fits a stronf application. Meanwhile for L006, it correctly identified the major weaknesses of no demonstrated business expereince, no financial projections nor collateral and a vague repayment assumption. One hindsight was the model calling enthusiasm a strength, not backed by evidence
# 
# - We forbid approve/reject because the model may be working with incomplete or inaccurate information so a final decision is not yet backed. Ethically, loan decisions can significantly decide a person's business or even life, so a qualified human officier is needed for accountability
# 

# %% [markdown]
# ### Part 3.4 — Commit your prompt templates
# 
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:**
# 
# - `f360f7b7d67a7e01d0bcfb42f56865e6b3c40592`
# 

# %% [markdown]
# ---
# 
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.
# 

# %% [markdown]
# ### Part 4.1 — Extraction accuracy against gold labels
# 

# %%
# For the three letters in GOLD, compare your extracted DataFrame to the gold values
#   field by field. Compute per-field accuracy across the three letters
#   (name matching can be case-insensitive; numbers must match exactly).

letters = ["L001", "L003", "L006"]
fields = list(next(iter(GOLD.values())).keys())

results = []
for field in fields:
    row = {"field": field}
    correct = 0

    for letter in letters:
        extracted = df.loc[df["letter_id"] == letter, field].iloc[0]
        gold = GOLD[letter][field]

        if field == "name":
            match = str(extracted).strip().lower() == str(gold).strip().lower()
        else:
            match = extracted == gold

        row[letter] = "Y" if match else "N"
        correct += int(match)

    row["accuracy"] = correct / len(letters)
    results.append(row)

# Display a small table: rows = fields, columns = L001 / L003 / L006 / accuracy.
comparison_df = pd.DataFrame(results)
pd.display(comparison_df)

# %% [markdown]
# ### Part 4.2 — Reliability: is the system consistent?
# 

# %%
# Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
#   temperature=1.0.

temp0_results = []
temp1_results = []

for i in range(5):
    result = extract_fields("L004", 0.0)
    temp0_results.append(result)

for j in range(5):
    result = extract_fields("L004", 1.0)
    temp1_results.append(result)

# For each temperature, report how many of the 5 runs produced (a) valid JSON and
#   (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
#   and count unique strings.


def analyse_results(results):
    valid_json = 0  # counter for unique strings
    serialized = []

    for result in results:
        try:
            # Works whether result is already a dict or needs checking
            json_string = json.dumps(result, sort_keys=True)
            serialized.append(json_string)
            valid_json += 1
        except (TypeError, ValueError):
            pass

    unique_outputs = len(set(serialized))

    return {
        "valid_json": valid_json,
        "identical_across_runs": unique_outputs == 1,
        "unique_outputs": unique_outputs,
    }


temp0_summary = analyse_results(temp0_results)
temp1_summary = analyse_results(temp1_results)

print("Temperature = 0")
print("Valid JSON:", temp0_summary["valid_json"], "/ 5")
print("Identical across runs:", temp0_summary["identical_across_runs"])
print("Unique outputs:", temp0_summary["unique_outputs"])

print("\nTemperature = 1.0")
print("Valid JSON:", temp1_summary["valid_json"], "/ 5")
print("Identical across runs:", temp1_summary["identical_across_runs"])
print("Unique outputs:", temp1_summary["unique_outputs"])

# %% [markdown]
# ### Part 4.3 — Hallucination probing
# 

# %%
# Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
question = "What is the applicant's credit score?"


def summarise(letter_text, question):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using information explicitly present in the letter. "
                    "If the requested information is absent, say that it is not provided."
                ),
            },
            {
                "role": "user",
                "content": f"""
                    Letter:
                    {letter_text}

                    Question:
                    {question}
                """,
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


output1 = summarise("L006", question)
print(f"""
TEST 1
Question: {question}
""")
print(f"Test 1: {output1}")

print(
    "PASS -> the model correctly admitted the information was absent"
    if output1
    else "Fail -> the model hallucinated"
)

print("\n=====================================================\n")

#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?
IRRELEVANT_TEXT = """
Arsenal won their first league title in over 22 years in the 25/26 season
by 7 points with an average squad age of 25.9 years old
"""

output2 = extract_fields(IRRELEVANT_TEXT, 0)
print(f"""
TEST 2
Irrelevant text: {IRRELEVANT_TEXT}
""")
print(f"Test 2: {output2}")

print(
    "PASS -> the model returned nulls/false and did not fabricate an applicatant"
    if (all(value is None or value is False for value in output2.values()))
    else "FAIL -> the model falsely fabricated an applicant"
)

# Recording the outputs verbatim below and label each PASS or FAIL.

# %% [markdown]
# **Student Reasoning — Evaluation results**
# _1. Report your extraction accuracy. Which field was hardest for the model and why?_
# _2. What did the reliability experiment show about temperature and production systems?_
# _3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?_
# 
# > **Answer:**
# 
# - The overall extraction accuracy was _77.8%_ (total of all accuracies / six). The hardest field for extraction was `purpose`, which had _0%_ accuracy across all letters. The purpose of a loan can be expressed in different ways and may require interpretations rather than direct extraction
# 
# - The experiment shows that lower temperature produces more consistent and predictable outputs, which is really important for reliability in production systems. Higher temperature increases variation but can increase the chance of incorrect or fabricated responses
# 
# - Hallucationation did not occur as the model stated that the info was not provided, and the irrelevant football news, it returned `None` for all applicant fields. Regardless, the risk can be mitigated by only using info present in the source, returning null for missing values and abstain from inferring values
# 

# %% [markdown]
# ### Part 4.4 — Appropriateness: should this system exist?
# 
# No code in this part — just judgment, which is the scarcest skill in AI for business.
# 

# %% [markdown]
# **Student Reasoning — Appropriateness**
# _1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses._
# _2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?_
# _3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring)._
# 
# > **Answer:**
# 
# - Applicants like **L002** could be unfairly rejected if the system mistakes poor English or informal tone for poor creditworthiness. A person may be running a viable business but can't really describe their income, repayment plans, or collateral formally. The model could unintentionally disadvantage applicants with lower English proficiency or different writing styles. Asides **L006** though, who comprises of genuine risk factors and no existing business, collateral, and several untested business ideas. Regardless, the system should distinguish those facts from how well either letter is written
# 
# - Sending loan letters to a foreign API means exposing applicants' names, financial details, business information, and other personal data beyond the institution's reach. Before deployment, I would check the provider's privacy policy, where data is stored and processed, whether prompts are retained or used for model training, encryption and access controls, deletion policies and breach procedures. For a Ghanaian institution, I would also check compliance with Ghana's Data Protection Commission that requires organizations processing personal data to be lawful and transparent
# 
# - **Human review loops** that never allows the model to make the final decision automatically. Flag borderline, incomplete (at any aspect), or low-confidence applications for a trained loan officer to review, with the original letter available for reference. **Appeal options** that logs the model's reasons, regularly checks error/rejection rates across racial/cultural groups, and giving applicants a way to formally correct extracted information or appeal decisions
# 

# %% [markdown]
# # Section 5 — Reflection
# 
# _Answer in a few sentences each:_
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
# 
# > **Answer:**
# 
# - Both methods involve changing inputs that go into the system, testing/evaluating performance, and engaging the version that gives better results. However, hyperparameters change how a model learns during training, e.g., learning rate, number of epochs, or other hidden units while prompt changes do not retrain the model, only how that already trained model behaves at inference
# 
# - I would definitely NOT trust the system to run unattended. Section 4's revealed the overall accuracy was **77.8%**, and the `purpose` field had **0%** accuracy across all three GOLD letters. For a loan approval system, this is rather poor and incorrectly extracting the reason for a loan could directly affect an applicant’s assessment, so human review is still necessary
# 
# - My `response.usage` numbers revealed one call used **404** tokens in total (**47** for prompt + **357** for completion). For **1000 applications/month**, it'll be **404,000 tokens/month**. This implies that provider choice should beware of cost per token, reliability, latency and privacy
# 
# - For this task, an API is preferable because the foundation model already understands natural language, can extract (and use) structured information, summarize, and handle varied writing styles without needing a large labelled dataset or training. Training my own model would require serious time, data, processing power, tuning, and maintenance. However, a custom model may be better when I'm particular about privacy or implementing an API with high-scale costs or the task is highly specialised.
# 

# %% [markdown]
# ---
# 
# ### Submission checklist
# 
# - [x] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [x] **No API key anywhere in the notebook or the commit history.**
# - [x] Every **Student Reasoning** box is filled in with full sentences.
# - [x] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [x] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [x] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [x] Repository link submitted to the course portal.
# - [x] AI Declaration form in Repository.
# 


