import os
import re
import pandas as pd
import ollama
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "train.csv"
OUTPUT_TEST_CSV = "Zero_Shot_Result.csv"
MODEL_NAME = "gemma3:4b"
SAMPLE_SIZE = 100

def get_zero_shot_prompt(essay_text):
    """Generates the zero-shot prompt for the LLM."""
    return f"""You are an expert academic evaluator. Grade the following student essay on a scale from 1 to 6, where 1 is the lowest quality and 6 is the highest quality.

Evaluation Criteria:
- Score 1: Poor organization, severe grammatical errors, and fails to address the topic.
- Score 2: Weak structure, frequent errors, and minimal development of ideas.
- Score 3: Satisfactory structure but lacks depth, with noticeable language errors.
- Score 4: Clear structure, well-developed arguments, and good command of language.
- Score 5: Strong organization, compelling arguments, and excellent vocabulary/grammar.
- Score 6: Exceptional essay with insightful ideas, sophisticated style, and flawless execution.

Essay to Grade:
\"\"\"
{essay_text}
\"\"\"

Provide your response in exactly the following format. Do not write any introduction, explanation, or justification. Just provide the final score.

Score: [Your score here, e.g., 4]"""

def extract_score(response_text):
    """Extracts the numerical score (1-6) from the LLM's response."""
    match = re.search(r'(?:score:\s*)?([1-6])', response_text.lower())
    if match:
        return int(match.group(1))
    return None

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Load full dataset
    df_full = pd.read_csv(INPUT_CSV)
    
    # Take a sample of 100 rows
    # Use .head(100) for the first 100 rows, or .sample(n=SAMPLE_SIZE, random_state=42) for a random mix
    df_sample = df_full.head(SAMPLE_SIZE).copy()
    print(f"Extracted {len(df_sample)} rows for testing.")

    predicted_scores = []

    # Iterate through the sample
    for index, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Testing 100 Essays"):
        essay_id = row['essay_id']
        full_text = row['full_text']
        
        prompt = get_zero_shot_prompt(full_text)
        
        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={
                    "temperature": 0.0,  # Deterministic
                }
            )
            
            raw_response = response['response'].strip()
            score = extract_score(raw_response)
            predicted_scores.append(score)
            
        except Exception as e:
            print(f"\nError processing essay {essay_id}: {e}")
            predicted_scores.append(None)

    # Add predictions to the sample dataframe
    df_sample['predicted_score'] = predicted_scores
    
    # Save test results
    df_sample.to_csv(OUTPUT_TEST_CSV, index=False)
    print(f"\nTesting complete! Results saved to {OUTPUT_TEST_CSV}")
    
    # Quick Summary Statistics if 'score' column exists
    if 'score' in df_sample.columns:
        # Filter out any None values if extraction failed
        valid_data = df_sample.dropna(subset=['predicted_score'])
        if not valid_data.empty:
            correct = (valid_data['score'] == valid_data['predicted_score']).sum()
            print(f"Exact Match Accuracy: {correct}/{len(valid_data)} ({correct/len(valid_data)*100:.2f}%)")

if __name__ == "__main__":
    main()
