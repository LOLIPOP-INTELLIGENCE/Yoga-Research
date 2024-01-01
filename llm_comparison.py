import requests
import json
import requests
import random
import base64
from PIL import Image
from io import BytesIO
import json
import cv2
import time
from google.cloud import texttospeech
from google.cloud import speech
from playsound import playsound

import os
import time

resource_name = "weev"
deployment_name = "DripVision"
api_key = "487dbbf0df454ee6be002a5f77b0d04f"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/blackhole/.config/gcloud/application_default_credentials.json'

client = texttospeech.TextToSpeechClient()
client_stt = speech.SpeechClient()

url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2023-12-01-preview"

all_poses = ['Sitting pose 1 (normal)', 'Side-Reclining_Leg_Lift_pose_or_Anantasana_', 'Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_', 'Pigeon_Pose_or_Kapotasana_', 'Plow_Pose_or_Halasana_', 'Bridge_Pose_or_Setu_Bandha_Sarvangasana_', 'Boat_Pose_or_Paripurna_Navasana_', 'Scorpion_pose_or_vrischikasana', 'Noose_Pose_or_Pasasana_', 'Garland_Pose_or_Malasana_', 'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_', 'Standing_Forward_Bend_pose_or_Uttanasana_', 'Plank_Pose_or_Kumbhakasana_', 'Camel_Pose_or_Ustrasana_', 'Virasana_or_Vajrasana', 'Frog_Pose_or_Bhekasana', 'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_', 'Half_Moon_Pose_or_Ardha_Chandrasana_', 'Low_Lunge_pose_or_Anjaneyasana_', 'Warrior_I_Pose_or_Virabhadrasana_I_', 'Rajakapotasana', 'Supported_Headstand_pose_or_Salamba_Sirsasana_', 'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_', 'Bound_Angle_Pose_or_Baddha_Konasana_', 'Fish_Pose_or_Matsyasana_', 'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II', 'Upward_Plank_Pose_or_Purvottanasana_', 'Dolphin_Pose_or_Ardha_Pincha_Mayurasana_', 'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_', 'Lord_of_the_Dance_Pose_or_Natarajasana_', 'Eagle_Pose_or_Garudasana_', 'Tortoise_Pose', 'Locust_Pose_or_Salabhasana_', 'Akarna_Dhanurasana', 'Yogic_sleep_pose', 'Handstand_pose_or_Adho_Mukha_Vrksasana_', 'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_', 'Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_', 'Tree_Pose_or_Vrksasana_', 'Heron_Pose_or_Krounchasana_', 'Gate_Pose_or_Parighasana_', 'Happy_Baby_Pose_or_Ananda_Balasana_', 'Cockerel_Pose', 'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_', 'Feathered_Peacock_Pose_or_Pincha_Mayurasana_', 'Intense_Side_Stretch_Pose_or_Parsvottanasana_', 'Split pose', 'Chair_Pose_or_Utkatasana_', 'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_', "Bharadvaja's_Twist_pose_or_Bharadvajasana_I_", 'Scale_Pose_or_Tolasana_', 'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana', 'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_', 'Warrior_II_Pose_or_Virabhadrasana_II_', 'Wind_Relieving_pose_or_Pawanmuktasana', 'Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_', 'Cat_Cow_Pose_or_Marjaryasana_', 'Child_Pose_or_Balasana_', 'Peacock_Pose_or_Mayurasana_', 'Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_', 'Crane_(Crow)_Pose_or_Bakasana_', 'desktop', 'Side_Plank_Pose_or_Vasisthasana_', 'Supta_Baddha_Konasana_', 'Shoulder-Pressing_Pose_or_Bhujapidasana_', 'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_', 'Staff_Pose_or_Dandasana_', 'Supta_Virasana_Vajrasana', 'viparita_virabhadrasana_or_reverse_warrior_pose', 'Cobra_Pose_or_Bhujangasana_', 'Firefly_Pose_or_Tittibhasana_', 'Extended_Puppy_Pose_or_Uttana_Shishosana_', 'Warrior_III_Pose_or_Virabhadrasana_III_', 'Corpse_Pose_or_Savasana_', 'Cow_Face_Pose_or_Gomukhasana_', 'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_', 'Legs-Up-the-Wall_Pose_or_Viparita_Karani_', 'Bow_Pose_or_Dhanurasana_', 'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_', 'Seated_Forward_Bend_pose_or_Paschimottanasana_', 'Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_', 'Wild_Thing_pose_or_Camatkarasana_', 'Eight-Angle_Pose_or_Astavakrasana_']

def image_to_base64(image_path):

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def capture_photo():
    time.sleep(5)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite('captured_photo.png', frame)
        print("Photo captured and saved!")
    else:
        print("Failed to capture photo")

    cap.release()

def get_random_images(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    urls = [line.split('\t')[1].strip() for line in lines]
    urls = [url for url in urls if url.startswith("http://allyogapositions.com")]
    random.shuffle(urls)

    valid_urls = []
    for url in urls:
        if len(valid_urls) == 3:
            break
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                if image.size[0] >= 500 and image.size[1] >= 500:
                    valid_urls.append(url)
        except Exception:
            continue

    return valid_urls if valid_urls else "No suitable image could be retrieved."

capture_photo()

image_path = "captured_photo.png"


headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

body = {
    "messages": [
        {
            "role": "system",
            "content": "You are Yoga Teacher."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"I will give you a list of Yoga poses and you will tell me which Yoga pose best matches the image. Your output should be a single json object with the key 'pose' and the value as the name of the pose from the list below. \n\n{all_poses}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_to_base64(image_path)}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 100,
    "stream": False
}


print('extracting yoga pose')
response = requests.post(url, headers=headers, data=json.dumps(body))


yoga_pose = json.loads(response.content)['choices'][0]['message']['content']
yoga_pose_extracted = yoga_pose[yoga_pose.find('"pose": "') + len('"pose": "') : yoga_pose.rfind('"')]
print(yoga_pose_extracted)

print('extracting url images')
urls = get_random_images(f'Yoga-82/yoga_dataset_links/{yoga_pose_extracted}.txt')
print(urls)


def url_to_base64(url):
    response = requests.get(url, timeout=3)
    if response.status_code == 200:
        encoded_string = base64.b64encode(response.content).decode('utf-8')
    return encoded_string

def create_body(yoga_pose_extracted, urls):
    new_body = {
        "messages": [
            {
                "role": "system",
                "content": "You are Yoga Teacher."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"I am attempting to do the yoga pose of {yoga_pose_extracted}. I need you to tell me if I am doing it correctly. To help you with this, I will give you images of professionals doing the pose. You should tell me corrective instructions only for my pose that will help me do the pose correctly. If you find no errors, then tell me that I am doing the pose correctly."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(image_path)}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "stream": False
    }

    for i, url in enumerate(urls):
        new_body["messages"].append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Professional pose {i+1}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{url_to_base64(url)}"
                    }
                }
            ]
        })

    return new_body

new_body = create_body(yoga_pose_extracted, urls)

print('sending request for correction')
response = requests.post(url, headers=headers, data=json.dumps(new_body))
correction = (json.loads(response.content)['choices'][0]['message']['content'])

def tts(text):
    input = texttospeech.SynthesisInput(text = text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name = "en-US-Neural2-C", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding="LINEAR16", speaking_rate=1.0, pitch=1.0
    )

    response = client.synthesize_speech(
        input=input, voice=voice, audio_config=audio_config
    )

    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
    
    playsound('output.wav', True)

tts(correction)