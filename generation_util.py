import re
import random
import torch
import openai
from ratelimiter import RateLimiter
from retrying import retry
import urllib.request

### --- ### 
# WARNING: Change the API setting according to your account
openai.api_key = YOUR_API_KEY
openai.organization = YOUR_ORGANIZATION
### --- ### 

# To avoid exceeding rate limit for ChatGPT API
@retry(stop_max_delay=3000, wait_fixed=1000)
@RateLimiter(max_calls=2000, period=60)
def generate_response_chatgpt(utt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", "content": utt}
        ]
        )
    return response.choices[0].message.content
    
def generate_response_fn(utt, model, pipe=None):
    if model == 'chatgpt':
        return generate_response_chatgpt(utt)
    elif model in ['llama3', 'mistral']:
        return pipe(utt, max_new_tokens=512)[0]['generated_text']