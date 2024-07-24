import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def get_subset_names():
    all_subsets = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
                   'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
                   'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
                   'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
                   'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
                   'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
                   'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
                   'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
                   'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
                   'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 
                   'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 
                   'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 
                   'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 
                   'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    return all_subsets

def load_model_and_tokenizer():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def load_data(subset):
    dataset = load_dataset("cais/mmlu", subset)
    return dataset['test']

class MMluDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = format_prompt(item['question'], item['choices'])
        return prompt, item['answer']

def format_prompt(question, choices):
    formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    return f"Answer the following multiple-choice question by responding with the letter of the correct answer (A, B, C, or D).\n\nQuestion: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"

def get_most_probable_answer(logits, tokenizer):
    last_token_logits = logits[:, -1, :]
    
    answer_logits = {}
    for letter in "ABCD":
        for prefix in ["", " "]:
            token_id = tokenizer.encode(f"{prefix}{letter}", add_special_tokens=False)[0]
            answer_logits[letter] = max(answer_logits.get(letter, float('-inf')), last_token_logits[:, token_id].item())
    
    return max(answer_logits, key=answer_logits.get)

def generate_batch(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = [get_most_probable_answer(outputs.logits[i:i+1], tokenizer) for i in range(len(prompts))]
    return results

def evaluate_model(model, tokenizer, test_set, batch_size):
    dataset = MMluDataset(test_set)
    total = len(test_set)
    results = []
    correct = 0
    
    progress_bar = tqdm(range(0, total, batch_size), desc="Evaluating")
    for i in progress_bar:
        batch_prompts = [dataset[j][0] for j in range(i, min(i + batch_size, total))]
        batch_answers = [dataset[j][1] for j in range(i, min(i + batch_size, total))]
        
        try:
            model_answers = generate_batch(model, tokenizer, batch_prompts)
            batch_results = [model_ans == chr(65 + correct_ans) for model_ans, correct_ans in zip(model_answers, batch_answers)]
            
            for prompt, model_ans, correct_ans, is_correct in zip(batch_prompts, model_answers, batch_answers, batch_results):
                results.append({
                    "prompt": prompt,
                    "model_answer": model_ans,
                    "correct_answer": chr(65 + correct_ans),
                    "is_correct": is_correct
                })
                if is_correct:
                    correct += 1
            
            progress_bar.set_postfix({"Accuracy": f"{correct / (i + len(batch_results)):.2%}"})
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error(f"CUDA out of memory. Try reducing batch size.")
                raise
            else:
                logger.error(f"Unexpected error: {e}")
                raise
        
        torch.cuda.empty_cache()
    
    final_accuracy = correct / total
    logger.info(f"\nFinal Accuracy: {final_accuracy:.2%}")
    return results, final_accuracy

def evaluate_subset(model, tokenizer, subset_name):
    logger.info(f"Evaluating subset: {subset_name}")
    test_set = load_data(subset_name)
    results, accuracy = evaluate_model(model, tokenizer, test_set, BATCH_SIZE)
    
    # Save results to file
    with open(f"results_{subset_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results for {subset_name} saved to results_{subset_name}.json")
    return accuracy

def save_overall_scores(scores):
    with open("overall_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    logger.info("Overall scores saved to overall_scores.json")

def main():
    torch.cuda.empty_cache()
    torch.random.manual_seed(0)
    
    model, tokenizer = load_model_and_tokenizer()
    subsets = get_subset_names()
    
    overall_scores = {}
    
    for subset in subsets:
        accuracy = evaluate_subset(model, tokenizer, subset)
        overall_scores[subset] = accuracy
    
    save_overall_scores(overall_scores)

if __name__ == "__main__":
    BATCH_SIZE = 8  # Adjust based on your GPU memory
    main()