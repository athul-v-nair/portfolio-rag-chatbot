import json
from src.evaluation.evaluator import RAGEvaluator

def evaluation_pipeline(dataset_path="src/evaluation/evaluation_dataset.json", top_k=3):
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    evaluator = RAGEvaluator()

    # Run evaluation
    results = evaluator.evaluate(dataset, top_k=top_k)

    # Summary
    summary = evaluator.summarize(results)

    # Pretty print
    print("\n--- Individual Results ---")
    for r in results:
        print(r)

    print("\n--- Summary ---")
    print(summary)

    return results, summary

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    evaluation_pipeline()