from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from yolov8_bubbles import * # bubble_detect(), bubble_on_panel(), text_on_bubble(), text_on_bubble_on_panel()
from yolov8_panel import *   # split_image(), panel_seg()
from clova_ocr import *      # image_ocr()
from gpt_captioning import * # image_captioning()
from assistants_api import * # new_assistant(), new_book(), assistant_image_captioning()
from s3_upload import *      # imread_url()


app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    # 파일이 있는지 체크
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    #파일 저장
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image_name = filename.rsplit('.', 1)[0]
        s3.upload_file(filepath,"meowyeokbucket",f"comics/Pages/{image_name}.jpg")
        
        texts = get_text(image_name)
        # print(texts)

        thread_id = new_book()
        # print(thread_id)

        messages = assistant_image_captioning(image_name, texts, assistant_id, thread_id)
        # print(messages)

        res = parse_texts(messages)
        
        return res, 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(host='0.0.0.0', port=5000)