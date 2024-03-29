import os
from tqdm import tqdm 
import json

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def analyze_image(image_path, prompt):
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=False, device=device)

    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    output = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1)
    return output
# Iterate over each image in the folder
results = []
instructions = [
    "widen his mouth let him looks shocked",
    "widen his mouth let her looks smile",
    "change to smile",
    "Change only the lips to red, without altering any other part of the face",
    "Add long eyelashes on eyes",
    "Big eyes",
    "Close the eyes",
    "Change the eye color to green",
    "Make him frown",
    "Wearing a hat",
    "Add a beard to the face",
    "Add glasses on the face",
    "wearing sunglasses",
    "Add a moustache",
    "Add earrings",
    "Add wrinkles to the face",
    "Add acne on the face",
    "Change hair color to blonde",
    "Make her looks older",
    "Face freckles",
    "Darken the skin tone",
    "Add a double chin",
    "Make skin paler",
    "Add blue eye shadow",
    "Add scar on the face",
    "Change lips to sad",
    "Make him terrifying",
    "Add dark circles under eyes",
    "Squint the eyes slightly",
    "Make eyebrows bushy",
    "Change to crying face",
    "Change to angry face"
]
output_jsonl = '/storage1/ruby/LAVIS/deepfake/output.jsonl'
root_path = '/storage1/ruby/instruct-pix2pix/ffhqtest'

for subfolder in ['fake', 'real']:
        folder_path = os.path.join(root_path, subfolder)            
        for filename in tqdm(os.listdir(folder_path)):
            try:
                img_path = os.path.join(folder_path, filename)
                suffix = filename.split('_')[-1].split('.')[0]
                try:
                    if subfolder=='real':
                        prompt = "Is this photo real( answer Yes or No)? If not, why?"
                    else:
                        index = int(suffix) - 1  # 转换为索引
                        hint = instructions[index]
                        prompt = f"Is this photo real( answer Yes or No)? If not, why? hint:fake,edition: {hint}"
                except ValueError:
                    prompt = "Is this photo real( answer Yes or No)? If not, why?"
                result = analyze_image(img_path, prompt)

                results.append({"filename": filename, "content": result})

                print("image: ", img_path,", reason: ", result)
            except Exception as e:
                print("image:",img_path," error:", e)
            
with open(output_jsonl, 'w') as outfile:
    for entry in results:
        json_line = json.dumps(entry)
        outfile.write(json_line + '\n')
        
    