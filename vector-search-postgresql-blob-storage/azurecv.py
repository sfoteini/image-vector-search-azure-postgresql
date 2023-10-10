import requests
import json

def image_embedding(image, endpoint, key):
    """
    Image embedding using Azure Computer Vision 4.0 (Vectorize Image API)
    """
    with open(image, 'rb') as img:
        data = img.read()

    # Image retrieval API
    version = "?api-version=2023-02-01-preview&modelVersion=latest"
    vectorize_img_url = endpoint + "retrieval:vectorizeImage" + version

    headers = {
        'Content-type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': key
    }

    try:
        r = requests.post(vectorize_img_url, data=data, headers=headers)

        if r.status_code == 200:
            image_vector = r.json()['vector']
            return image_vector
        else:
            print(f"An error occurred while processing {image}. Error code: {r.status_code}.")
        
    except Exception as e:
        print(f"An error occurred while processing {image}: {e}")

    return None

def text_embedding(prompt, endpoint, key):
    """
    Text embedding using Azure Computer Vision 4.0 (Vectorize Text API)
    """
    text = {'text': prompt}

    # Image retrieval API
    version = "?api-version=2023-02-01-preview&modelVersion=latest"
    vectorize_txt_url = endpoint + "retrieval:vectorizeText" + version

    headers = {
        'Content-type': 'application/json',
        'Ocp-Apim-Subscription-Key': key
    }

    try:
        r = requests.post(vectorize_txt_url, data=json.dumps(text), headers=headers)

        if r.status_code == 200:
            text_vector = r.json()['vector']
            return text_vector
        else:
            print(f"An error occurred while processing the prompt '{text}'. Error code: {r.status_code}.")
        
    except Exception as e:
        print(f"An error occurred while processing the prompt '{text}': {e}")
    
    return None