import base64
import requests
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.environ.get("openai_key")

def image_captioning(name, texts, index) :
    
    # Path to your image
    image_url = f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/panel_seg/{name}/{name}_{index}.jpg"
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_key}"
    }
    
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user", 
                "content": f"""
                    Please convert the following texts into a script format as dialogue between characters.
                    Also, briefly describe the scene in the image in two lines based on this dialogue, 'description' first, then 'dialogue':
    
                    Texts (speech bubbles): {texts}
                    Image: {image_url}
                    """
                }
        ],
        "max_tokens": 200 
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    pretty_response = json.dumps(response.json(), ensure_ascii=False, indent=4, sort_keys=True)
    pretty_response_with_newlines = re.sub(r'\\n|(?<!")\\', lambda match: '\n' if match.group(0) == '\\n' else '', pretty_response)
    
    # Print the modified response
    print(pretty_response_with_newlines)

