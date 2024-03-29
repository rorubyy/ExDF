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
        "Which answer more accurately identifies the deepfake part of the image?\n\n Answer 1: Yes. Upon visual analysis, the beard appears digitally manipulated. The hair texture looks unnatural and inconsistent with surrounding facial features. The beard's edges are too sharp, and it doesn't blend seamlessly with the skin.}\n\n Answer 2: yes upon analyzing the image, the beard appears unnatural and inconsistent with the subject's facial features the beard's texture and color do not blend seamlessly with the subject's skin, and the hair"
    )
    review = get_eval(content, 50)
    scores = parse_score(review)


    # f_q = open(os.path.expanduser(args.question))
    # f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    # f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    # rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    # if os.path.isfile(os.path.expanduser(args.output)):
    #     cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    # else:
    #     cur_reviews = []

    # review_file = open(f'{args.output}', 'a')

    # context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    # image_to_context = {context['image']: context for context in context_list}

    # handles = []
    # idx = 0
    # for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
    #     ques = json.loads(ques_js)
    #     ans1 = json.loads(ans1_js)
    #     ans2 = json.loads(ans2_js)

    #     inst = image_to_context[ques['image']]
    #     cap_str = '\n'.join(inst['captions'])
    #     box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

    #     category = json.loads(ques_js)['category']
    #     if category in rule_dict:
    #         rule = rule_dict[category]
    #     else:
    #         assert False, f"Visual QA category not found in rule file: {category}."
    #     prompt = rule['prompt']
    #     role = rule['role']
    #     content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
    #                f'[Question]\n{ques["text"]}\n\n'
    #                f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
    #                f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
    #                f'[System]\n{prompt}\n\n')
    #     cur_js = {
    #         'id': idx+1,
    #         'question_id': ques['question_id'],
    #         'answer1_id': ans1.get('answer_id', ans1['question_id']),
    #         'answer2_id': ans2.get('answer_id', ans2['answer_id']),
    #         'category': category
    #     }
    #     if idx >= len(cur_reviews):
    #         review = get_eval(content, args.max_tokens)
    #         scores = parse_score(review)
    #         cur_js['content'] = review
    #         cur_js['tuple'] = scores
    #         review_file.write(json.dumps(cur_js) + '\n')
    #         review_file.flush()
    #     else:
    #         print(f'Skipping {idx} as we already have it.')
    #     idx += 1
    #     print(idx)
    # review_file.close()

