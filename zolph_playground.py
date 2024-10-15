#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
from yolov8_bubbles import * # bubble_detect(), bubble_on_panel(), text_on_bubble(), text_on_bubble_on_panel()
from yolov8_panel import *   # split_image(), panel_seg(), get_text(), parse_texts()
from clova_ocr import *      # image_ocr()
from gpt_captioning import * # image_captioning()
from assistants_api import * # new_assistant(), new_book(), assistant_image_captioning()
from s3_upload import *      # imread_url()


# In[2]:


# split_image('greece_1')
# panel_seg('greece_1_left')
# bubble_detect('greece_1_left')


# In[3]:


# texts = get_text('comic1_left') # 2차원 리스트 형태로 반환
# texts = [['Oh right! My new assistant also needs to be willing to carry my books and other supplies, extensively documentt and organize our ever growing catalog...'], ['...sort and bag back issues dust, clean, and maybe even some light construction work.', 'And listen to her unfunny stories!'], ["If all that sounds good to you, let's talk compensation! The reward for being my assistant, besides title, the is a highly chance respected to partake in my vast knowledge of comics."], ["Consider it a master's class in how to make comics!", 'She actually knows quite a lot!'], ['So what do you say? Are you willing to help US rebuild the library?'], [], ["Great! Let's get started right away!"]]
# print(texts)


# In[4]:


# thread_id = new_book()
# print(thread_id)


# In[5]:


# messages = assistant_image_captioning('comic1_left', texts, assistant_id, thread_id)
# print(messages)


# In[62]:


import json
import re

def parse_texts(texts):
    def parse_dialogue(dialogue_text):
        dialogues = []
        
        # Adjusted pattern to match dialogue and narration
        pattern = re.compile(r"\*\*(.+?)\:\*\*(.+?)(?=\s*\*\*|$)", re.DOTALL)
        matches = pattern.findall(dialogue_text)

        for match in matches:
            character = match[0].strip()  # Character name
            line = match[1].strip().replace('\n', ' ')  # Dialogue text
            dialogues.append({character: line})
        
        return dialogues
    
    script_json = []

    for script in texts:
        # Check if the script is a list (nested structure) and get the first element
        if isinstance(script, list):
            script = script[0]  # Take the first string from the nested list
        
        # Directly treat the script as a string
        scene = script.strip()  # Each script is now a string
        
        # Remove newlines in scene description and split by **Dialogue:** if it exists
        scene_parts = scene.replace('\n', ' ').split('**Dialogue:**')
        description = scene_parts[0].replace("**Scene Description:**", "").strip()
        
        # Only parse dialogue if it's present
        dialogues = parse_dialogue(scene_parts[1].strip()) if len(scene_parts) > 1 else []
        script_json.append({description: dialogues})
    
    # Output as JSON formatted string without extra escape characters
    json_output = json.dumps(script_json, indent=4, ensure_ascii=False)
    
    return json_output


# In[63]:


texts=[["In a dramatic outdoor scene, three characters are engaged in a tense confrontation. A well-dressed man, presumably Bart, looks sternly at a couple. The woman, Eileen, is caught between the two men. The man with Eileen, Tony, seems assertive, while she appears conflicted and remorseful.\n\n**Tony:** Get away from her, Bart! She's with me!\n\n**Bart:** You're not his girl, Eileen! Tell him!\n\n**Eileen (narrating, internally):** For one wild ecstatic moment, I would have gone anywhere with him...\n\n**Eileen:** I'm sorry, Bart! I shouldn't have done this! I... I'd like to come back... if it's all right."], ["In a tense exchange set in front of a grand building, Eileen, wearing a red dress, stands between two formally dressed men, Tony and Bart. Her expression is earnest and apologetic as she reaches out toward Bart, who looks stern yet composed. Tony gestures assertively, indicating his agitation.\n\n**Eileen:** I'm sorry, Bart! I shouldn't have done this! I... I'd like to come back... if it's all right.\n\n**Bart:** I... All right, Eileen! It's all right!"], ["The panel contains a reflective narrative text set against a plain background, providing a backstory with hints of nostalgia and change. It reflects Eileen's thoughts about the past events involving Tony and Bart, describing their paths and her lingering emotions.\n\n**Eileen (narration):** After that night, Tony seemed to vanish from the face of the Earth, that fall. I took a job as a secretary, and Bart went off to college. He wrote often, and I saw him during vacations and on occasional weekends, when he could come down. A year passed, and then part of another—and then one night... Even if he was a little drunk, I was glad to see him. The minute he came in, the old feeling began to come back..."], ["In a cozy indoor setting, Eileen, wearing a red dress, stands at the door, surprised and welcoming as Tony, in a blue suit and patterned tie, appears at her doorstep with a friendly and slightly tipsy demeanor. Tony gestures enthusiastically, indicating his relaxed and cheerful mood.\n\n**Eileen:** Tony! I... come in!\n\n**Tony:** Hello, baby! Just thought I'd look up my old friend... It's been a long time... wondered how you were!"], ["In a warmly lit room, Eileen, with an elegant hairstyle and wearing pearls, expresses a mix of curiosity and slight reproach as she converses with Tony. Tony, still projecting a carefree and slightly tipsy attitude, stands with a confident smile, holding a drink, and compliments her appearance.\n\n**Eileen:** You could have found out how I was, Tony! I've been right here! What about you? What have you been doing?\n\n**Tony:** Don't worry about old Tony, baby! I can take care of myself! Let's talk about you... You look like a million bucks!"], ["In a close and intimate moment, Tony leans in with a smirk, attempting to kiss Eileen. Eileen appears startled and apprehensive, pulling back slightly. Her expression reveals internal conflict, suggesting a mix of lingering affection and resistance to Tony's advance.\n\n**Tony:** I've missed you, baby! Gimme a little kiss... just for ol' times' sake—how about it?\n\n**Eileen:** Don't, Tony! Please! I mean it!"], ["In a heated moment, Eileen, looking determined and assertive, pushes Tony away with her hands as he recoils, rubbing his cheek with a surprised expression. The tension between them is palpable, as Eileen's firm stance contrasts with Tony's startled realization.\n\n**Eileen:** I mean it, Tony! Take your hands off me!\n\n**Tony:** Oww! Sorry, baby..."]]
res = parse_texts(texts)
print(res)


# In[ ]:





# In[ ]:




