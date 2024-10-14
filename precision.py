import json
import pandas as pd

# Load the prediction and ground truth files
with open('pred_retrieve.json', 'r') as pred_file:
    pred_retrieve = json.load(pred_file)

with open('ground_truths_example.json', 'r') as truth_file:
    ground_truths = json.load(truth_file)

# Convert to dictionaries for easy lookup
pred_answers = {item["qid"]: item["retrieve"] for item in pred_retrieve["answers"]}
ground_truths_dict = {item["qid"]: {"retrieve": item["retrieve"], "category": item["category"]} for item in ground_truths["ground_truths"]}

# Prepare the data for output
output_data = []
correct_predictions = 0  # Variable to count correct predictions
for qid, pred_retrieve_value in pred_answers.items():
    if qid in ground_truths_dict:
        ground_truth = ground_truths_dict[qid]
        is_correct = pred_retrieve_value == ground_truth["retrieve"]
        if is_correct:
            correct_predictions += 1
        output_data.append({
            "qid": qid,
            "predicted_retrieve": pred_retrieve_value,
            "ground_truth_retrieve": ground_truth["retrieve"],
            "ground_truth_category": ground_truth["category"],
            "is_correct": is_correct  # Add a column to indicate if the prediction was correct
        })

# Calculate Precision@1
total_questions = len(ground_truths_dict)
precision_at_1 = correct_predictions / total_questions if total_questions > 0 else 0

# Print the precision in the log
print(f'Precision@1: {precision_at_1:.7f}')

# Create a pandas DataFrame
df = pd.DataFrame(output_data)

# Save to Excel
output_excel_path = 'pred_vs_ground_truth_1014.xlsx'
df.to_excel(output_excel_path, index=False)

print(f'Excel file saved to: {output_excel_path}')
