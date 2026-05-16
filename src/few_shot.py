import pandas as pd
import requests
import re
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "llama3"
NUM_SAMPLES = 100

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("data/train.csv")

df = df.head(NUM_SAMPLES)

# =========================================================
# FEW SHOT EXAMPLES
# =========================================================

# ambil contoh tetap
few_shot_df = df.sample(3, random_state=42)

few_shot_examples = ""

for _, row in few_shot_df.iterrows():

    few_shot_examples += f"""
Essay:
{row['full_text']}

Score:
{row['score']}

"""

# =========================================================
# FUNCTION ASK LLM
# =========================================================

def ask_llm(essay):

    prompt = f"""
You are a professional English essay grader.

Below are examples of essays and their scores.

{few_shot_examples}

Now evaluate the next essay.

Evaluation criteria:
- grammar
- coherence
- vocabulary
- clarity
- structure

Scoring rubric:
1 = very poor
2 = poor
3 = average
4 = good
5 = very good
6 = excellent

You MUST return ONLY ONE integer number from 1 to 6.

Do not explain.
Do not output text.
Do not output sentences.

Essay:
{essay}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
    )

    result = response.json()["response"].strip()

    return result

# =========================================================
# PARSE SCORE
# =========================================================

def parse_score(text):

    match = re.search(r"[1-6]", text)

    if match:
        return int(match.group())

    return 3

# =========================================================
# INFERENCE
# =========================================================

predictions = []

for essay in tqdm(df["full_text"]):

    try:

        raw_output = ask_llm(essay)

        score = parse_score(raw_output)

    except Exception as e:

        print("ERROR:", e)

        score = 3

    predictions.append(score)

# =========================================================
# SAVE RESULT
# =========================================================

df["predicted_score"] = predictions

df.to_csv(
    "outputs/few_shot_result.csv",
    index=False
)

print("\n===== FEW SHOT RESULT =====")
print(df[["score", "predicted_score"]].head())