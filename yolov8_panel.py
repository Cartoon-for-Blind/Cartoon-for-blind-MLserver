import os
import cv2
import json
import re
import matplotlib.pyplot as plt
from ultralytics import YOLO
from yolov8_bubbles import * # bubble_detect(), bubble_on_panel(), text_on_bubble(), text_on_bubble_on_panel()
from s3_upload import *      # imread_url()
from clova_ocr import *      # image_ocr()


def split_image(name):
    image = imread_url(f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/Pages/{name}.jpg")
    height, width, _ = image.shape
    mid_width = width // 2
    left_image = image[:, :mid_width]   # 왼쪽 절반
    right_image = image[:, mid_width:]  # 오른쪽 절반
    
    # BGR에서 RGB로 변환 (Matplotlib에서 색상 맞추기 위해)
    left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    
    # 나눈 이미지를 표시
    plt.figure(figsize=(10, 5))
    
    # 좌측 이미지 출력
    plt.subplot(1, 2, 1)
    plt.imshow(left_image_rgb)
    plt.title('Left Half')
    plt.axis('off')
    
    # 우측 이미지 출력
    plt.subplot(1, 2, 2)
    plt.imshow(right_image_rgb)
    plt.title('Right Half')
    plt.axis('off')
    
    # 출력
    plt.show()
       
    # 좌측,우측 이미지 저장 경로
    left_image_path = (f'C:\\Users\\vkdnj\\Zolph\\comics\\Pages\\{name}_left.jpg')
    right_image_path = (f'C:\\Users\\vkdnj\\Zolph\\comics\\Pages\\{name}_right.jpg')
    
    # 좌측 및 우측 이미지 저장
    cv2.imwrite(left_image_path, left_image)
    cv2.imwrite(right_image_path, right_image)
    # s3에도 업로드
    s3.upload_file(left_image_path,"meowyeokbucket",f"comics/Pages/{name}_left.jpg")
    s3.upload_file(right_image_path,"meowyeokbucket",f"comics/Pages/{name}_right.jpg")


def sort_panels(boxes_with_objects, y_threshold=20):

    # 먼저 y1 값을 기준으로 오름차순 정렬
    boxes_with_objects.sort(key=lambda item: item[0][1])

    # 비슷한 y 값 그룹으로 나누기
    grouped_boxes = []
    current_group = [boxes_with_objects[0]]
    
    for i in range(1, len(boxes_with_objects)):
        _, y1_current, _, _ = boxes_with_objects[i][0]
        _, y1_last, _, _ = boxes_with_objects[i - 1][0]
        
        # y 값이 임계값 이내라면 같은 그룹에 추가
        if abs(y1_current - y1_last) <= y_threshold:
            current_group.append(boxes_with_objects[i])
        else:
            # 새로운 그룹 생성
            grouped_boxes.append(current_group)
            current_group = [boxes_with_objects[i]]
    
    # 마지막 그룹 추가
    if current_group:
        grouped_boxes.append(current_group)
    
    # 각 그룹 내에서 x1 값을 기준으로 정렬
    for group in grouped_boxes:
        group.sort(key=lambda item: item[0][0])  # x1을 기준으로 정렬

    # 그룹을 flatten하여 반환
    sorted_boxes = [item for group in grouped_boxes for item in group]
    return sorted_boxes

def panel_seg(name, y_threshold=20):

    # Load a model
    model = YOLO('C:\\Users\\vkdnj\\Zolph\\models\\panel_best.pt') 
    
    # Load an image
    image = imread_url(f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/Pages/{name}.jpg")
    # Run inference
    results = model(image, conf=0.4)
    
    # 결과에서 탐지된 패널들의 좌표를 저장할 리스트
    boxes_with_objects = []  # 박스 좌표와 이미지를 저장할 리스트

    # 탐지된 객체들에 대한 정보를 추출하여 좌표 리스트 생성
    for idx, box in enumerate(results[0].boxes.xyxy):  # xyxy 포맷의 바운딩 박스 좌표
        x1, y1, x2, y2 = map(int, box)
        detected_object = results[0].orig_img[y1:y2, x1:x2]
        # 좌표와 이미지를 함께 저장
        boxes_with_objects.append(((x1, y1, x2, y2), detected_object))

    # 박스 정렬 (패널들을 y 값을 기준으로 정렬)
    sorted_boxes = sort_panels(boxes_with_objects, y_threshold)
    # 정렬된 좌표 리스트를 생성
    panel_coords_list = [(box[0][0], box[0][1], box[0][2], box[0][3]) for box in sorted_boxes]

    # 저장 경로 설정 및 폴더 생성 (없으면 생성)
    folder_path = f"C:\\Users\\vkdnj\\Zolph\\panel_seg\\{name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 정렬된 패널 이미지를 저장
    for idx, (box, detected_object) in enumerate(sorted_boxes):
        file_path = os.path.join(folder_path, f"{name}_{idx}.jpg")
        cv2.imwrite(file_path, detected_object)

    # 결과 표시
    result_plotted = results[0].plot()
    
    # 원본 이미지에 탐지 결과를 덮어쓰고 저장
    cv2.imwrite(f"{folder_path}\\{name}.jpg", result_plotted)
    # s3에도 업로드
    s3.upload_file(f"{folder_path}\\{name}.jpg","meowyeokbucket",f"comics/panel_seg/{name}/{name}.jpg")
    
    # 정렬된 패널 좌표 리스트를 반환
    return panel_coords_list


def get_text(name):
    panel_coords_list = panel_seg(name)
    bubble_coords_list = bubble_detect(name)
    text_coords_list = image_ocr(name)
    
    classified_bubbles = bubble_on_panel(bubble_coords_list, panel_coords_list)
    classified_texts = text_on_bubble(bubble_coords_list, text_coords_list)
    panel_texts = text_on_bubble_on_panel(classified_bubbles, classified_texts)
    
    texts = []
    for outer_key in sorted(panel_texts.keys()):
        inner_list = []
        for inner_key in sorted(panel_texts[outer_key].keys()):
            if panel_texts[outer_key][inner_key]:  # 비어 있지 않은 경우에만 추가
                inner_list.append(" ".join(panel_texts[outer_key][inner_key]))
        texts.append(inner_list)

    return texts



def parse_texts(texts):
    def parse_dialogue(dialogue_text):
        dialogues = []
        
        pattern = re.compile(r"\*\*(.+?):\*\*(.+)")
        matches = pattern.findall(dialogue_text)
        
        for match in matches:
            character = match[0].strip()  # Character name
            line = match[1].strip()      # Dialogue text
            dialogues.append({character : line})
        
        return dialogues
    
    script_json = []

    for script in texts:
        scene = script[0]
        
        scene_parts = scene.split('**Dialogue:**')
        description = scene_parts[0].replace("**Scene Description:**", "").strip()
        
        dialogues = parse_dialogue(scene_parts[1].strip()) if len(scene_parts) > 1 else []
        script_json.append({ description : dialogues })
    
    # Output as JSON formatted string
    json_output = json.dumps(script_json, indent=4, ensure_ascii=False)

    return json_output