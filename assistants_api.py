import os
from openai import OpenAI
import time
import requests
from dotenv import load_dotenv

load_dotenv()
openai_key = os.environ.get("openai_key")

API_KEY = openai_key
client = OpenAI(api_key = API_KEY)

assistant_id = 'asst_ZXVNifneCgUPLOjlYmJaNujO'


def new_assistant():
    assistant = client.beta.assistants.create(
       name="comics describer",
       instructions="""
       You are an assistant that takes speech bubbles from comic book images and converts them into a script format as dialogue between characters. 
       Along with the dialogue, provide a very short, simple description of the scene depicted in the image, focusing on visual elements like characters, actions, and expressions.
       The description should be concise and follow this format: first the description, then the dialogue..
       """,
       model="gpt-4o"
    )
    return assistant.id


def new_book():    
    #책마다 최초1회 부여해놓고 저장해서 이어 읽을때 써야함
    thread = client.beta.threads.create()
    return thread.id


def fetch_image_url(name, index):
    """Generate the image URL based on the provided name and index."""
    return f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/panel_seg/{name}/{name}_{index}.jpg"

def image_exists(image_url):
    """Check if the image exists by sending a HEAD request."""
    response = requests.head(image_url)
    return response.status_code == 200

def create_message_content(image_url, texts_str):
    """Create the content for the message based on the image URL and text."""
    content = [{
        "type": "image_url",
        "image_url": {"url": image_url, "detail": "low"}
    }]
    
    if texts_str.strip():  # Check if the text string is not empty
        content.append({
            "type": "text",
            "text": texts_str  # Add the text
        })

    return content

def wait_for_run_completion(run):
    """Wait until the run is completed, handling any failures."""
    while run.status != 'completed':
        time.sleep(0.1)  # Sleep briefly before polling again
        run = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )
        if run.status == 'failed':
            print("Run failed with error:", run.last_error)
            return None  # Indicate failure

    return run  # Return the completed run


def collect_messages(messages, target_run_id):
    """Collect messages in a structured format where run_id matches target_run_id."""
    collected_messages = []  # List to hold collected messages
    for message in messages:
        role = message.role
        content_list = message.content
        if message.run_id == target_run_id and role == 'assistant':  # Check run_id match
            message_texts = []  # List to hold texts for the current message
            for content in content_list:
                if hasattr(content, 'text'):  # Check if content has text attribute
                    message_texts.append(content.text.value)
            if message_texts:
                collected_messages.append(message_texts)  # Append the list of texts to collected_messages
    return collected_messages


def assistant_image_captioning(name, texts, assistant_id, thread_id):
    """Main function to handle image captioning with the assistant."""
    all_collected_messages = []  # Store collected messages

    image_index = 0
    while True: 
        image_url = fetch_image_url(name, image_index)

        if not image_exists(image_url): 
            break

        texts_str = "\",\"".join(["".join(text) for text in texts[image_index]]) if image_index < len(texts) else ""

        content = create_message_content(image_url, texts_str)

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id 
        )

        run = wait_for_run_completion(run)
        if run is None:  # Check if run failed
            break
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            collected_messages = collect_messages(messages.data, run.id)  # Collect messages
            all_collected_messages.extend(collected_messages)  # Add to all_collected_messages
        
        image_index += 1

    return all_collected_messages  # Return all collected messages

