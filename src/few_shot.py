import os
import re
import pandas as pd
import ollama
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "train.csv"
OUTPUT_TEST_CSV = "Few_Shot_Result.csv"
MODEL_NAME = "gemma3:4b"
INFERENCE_SAMPLES = 100
FEW_SHOT_SAMPLES = 8

def generate_few_shot_prefix(df_pool):
    """
    Randomly selects 8 essays from the pool to act as examples.
    Keeps the entire essay intact without stripping or truncating.
    """
    # Grab 8 random rows from the pool
    samples = df_pool.sample(n=FEW_SHOT_SAMPLES, random_state=42)
    
    prefix = "--- HUMAN GRADING EXAMPLES (FULL TEXT) ---\n"
    prefix += "To help calibrate your grading, here are 8 complete, real student essays and the FINAL SCORE they received from human graders:\n\n"
    
    for _, row in samples.iterrows():
        # Keep the full text completely intact
        full_example_text = str(row['full_text'])
        known_score = row['score']
        
        prefix += f"========== EXAMPLE OF A SCORE {known_score} ==========\n"
        prefix += f"\"{full_example_text}\"\n"
        prefix += f"Final Human Score: {known_score}\n"
        prefix += "================================================\n\n"
        
    return prefix

def get_hybrid_prompt(few_shot_prefix, essay_text):
    """Combines the 8 full-text examples with the aspect-based rubric for the target essay."""
    return f"""You are an expert academic evaluator. Analyze the following student essay by scoring three distinct dimensions on a scale from 1 to 6.

1. STRUCTURE (Organization & Flow)
   - Score 3: Basic paragraphs exist, but transitions are abrupt or formulaic.
   - Score 4: Logical progression of ideas with smooth transitions between paragraphs.
   - Score 5+: Mastery of pacing, with complex internal paragraph structure.

2. GRAMMAR (Mechanics & Vocabulary)
   - Score 3: Noticeable errors that do not completely obscure meaning; basic vocabulary.
   - Score 4: Few minor errors; demonstrates a varied and appropriate vocabulary.
   - Score 5+: Near flawless execution with highly sophisticated word choices.

3. DEVELOPMENT (Depth & Argumentation)
   - Score 3: Addresses the prompt but relies on superficial or repetitive claims.
   - Score 4: Explores the topic deeply with specific, relevant supporting details.
   - Score 5+: Nuanced argumentation that handles counterpoints gracefully.

{few_shot_prefix}

Now, read the FULL target essay below. Keeping the human grading examples and the rubric in mind, evaluate the essay.

Target Essay to Grade:
\"\"\"
{essay_text}
\"\"\"

Analyze the essay carefully. At the very end of your response, output your three scores wrapped strictly in the following XML tags:
<structure>X</structure>
<grammar>X</grammar>
<development>X</development>
(where X is an integer from 1 to 6)."""

# def calculate_final_score(response_text):
#     """Extracts the XML scores and averages them."""
#     try:
#         structure = int(re.search(r'<structure>\s*([1-6])\s*</structure>', response_text.lower()).group(1))
#         grammar = int(re.search(r'<grammar>\s*([1-6])\s*</grammar>', response_text.lower()).group(1))
#         development = int(re.search(r'<development>\s*([1-6])\s*</development>', response_text.lower()).group(1))
        
#         # Standard average (33% weight each)
#         average_score = (structure + grammar + development) / 3.0
#         final_score = int(average_score + 0.5) 
        
#         return final_score, structure, grammar, development
#     except (AttributeError, ValueError):
#         return None, None, None, None

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # 1. Load the full dataset
    df_full = pd.read_csv(INPUT_CSV)
    
    # 2. Split the data: Take 100 random samples for testing
    df_test = df_full.sample(n=INFERENCE_SAMPLES, random_state=99).copy()
    
    # 3. Create a pool of the REMAINING essays to draw our 8 examples from 
    df_pool = df_full.drop(df_test.index)
    
    few_shot_prefix = generate_few_shot_prefix(df_pool)
    
    predicted_scores = []
    structure_scores = []
    grammar_scores = []
    dev_scores = []

    for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Grading"):
        essay_id = row['essay_id']
        full_text = row['full_text']
        
        prompt = get_hybrid_prompt(few_shot_prefix, full_text)
        
        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={
                    "temperature": 0.0, 
                    # CRITICAL: Increased to 16384 to prevent Ollama from cutting off the 8 full essays!
                    "num_ctx": 16384 
                }
            )
            
            raw_response = response['response'].strip()
            final, struct, gram, dev = calculate_final_score(raw_response)
            
            predicted_scores.append(final)
            structure_scores.append(struct)
            grammar_scores.append(gram)
            dev_scores.append(dev)
            
        except Exception as e:
            print(f"\nError processing essay {essay_id}: {e}")
            predicted_scores.append(None)
            structure_scores.append(None)
            grammar_scores.append(None)
            dev_scores.append(None)

    # Save results
    df_test['predicted_score'] = predicted_scores
    df_test['score_structure'] = structure_scores
    df_test['score_grammar'] = grammar_scores
    df_test['score_development'] = dev_scores
    
    df_test.to_csv(OUTPUT_TEST_CSV, index=False)
    print(f"\nProcess complete saved to {OUTPUT_TEST_CSV}")
    
    # Calculate Accuracy
    if 'score' in df_test.columns:
        valid_data = df_test.dropna(subset=['predicted_score'])
        if not valid_data.empty:
            correct = (valid_data['score'] == valid_data['predicted_score']).sum()
            accuracy = (correct / len(valid_data)) * 100
            print(f"Exact Match Accuracy: {correct}/{len(valid_data)} ({accuracy:.2f}%)")
            
            close_calls = (abs(valid_data['score'] - valid_data['predicted_score']) <= 1).sum()
            close_accuracy = (close_calls / len(valid_data)) * 100
            print(f"Err +- 1 Point: {close_calls}/{len(valid_data)} ({close_accuracy:.2f}%)")

if __name__ == "__main__":
    main()
