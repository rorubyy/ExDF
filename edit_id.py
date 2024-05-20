import os
import json

def reassign_question_ids(directory):
    question_id_counter = 11795
    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        if filename == 'iDiff.json':
        # if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Check if 'question_id' is in the JSON and reassign it
            for item in data:
                if 'question_id' in item:
                    item['question_id'] = question_id_counter
                    question_id_counter += 1
                
            # Write the updated data back to the JSON file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

# Directory containing the JSON files
directory_path = '/storage1/ruby/LAVIS/lavis/output/5'

reassign_question_ids(directory_path)
print("Question IDs reassigned successfully.")
