import openai
from ratelimiter import RateLimiter
from retrying import retry

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
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": utt}
        ]
        )
    return response.choices[0].message.content
