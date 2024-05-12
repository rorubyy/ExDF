import argparse
import json
import os
from openai import AzureOpenAI
import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5

client = AzureOpenAI(api_key="4a95e5d7c6bb49198459fea94c289fd3",
api_version="2023-03-15-preview",
azure_endpoint="https://openai-ntust-demo.openai.azure.com/")
model_name = "gpt-4"

def get_eval(content: str, max_tokens: int):
    try:
        response = client.chat.completions.create(
            model= model_name,
            messages=[{
                'role': 'system',
                'content': "Based on the provided descriptions of an image, evaluate whether they indicate that the image is a deepfake and if they point to the same altered part of the image. Step 1: Determine if both descriptions suggest that the image is a deepfake. Answer with 'Yes' if both descriptions indicate the image is a deepfake; otherwise, answer with 'No'. Step 2: Only if both descriptions are identified as deepfake, then evaluate whether they point to the same altered part of the image, even if the details are not exactly the same. Consider them consistent if they both suggest the same part has been altered unnaturally, indicating the manipulation is in the same area. If the answer to Step 1 is 'No', then the evaluation for Step 2 should automatically be 'Inconsistent', as it implies the descriptions do not concur on the image being a deepfake. Please provide your answers based on the given criteria. Your responses should be accurate and reflect a careful evaluation of the descriptions provided. For Step 1, provide a 'Yes' or 'No'. For Step 2, provide 'Consistent' if the evaluation proceeds and the descriptions agree on the same altered part, or 'Inconsistent' if Step 1's answer is 'No' or the descriptions do not agree."
            }, {
                'role': 'user',
                'content': content,
            }],
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(e)

    return response.choices[0].message.content

def parse_score(review):
    try:
        split_text = review.split('\n')
        step1_result = split_text[0].split(': ')[1]
        step2_result = split_text[1].split(': ')[1]
        return [step1_result, step2_result]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    content = (
        "Which answer more accurately identifies the deepfake part of the image?`"
    )
    review = get_eval(content, 50)
    scores = parse_score(review)

