import os
import requests
import json
import math
from PIL import Image
import matplotlib.pyplot as plt
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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

def get_cosine_similarity(vector1, vector2):
    """
    Get the cosine similarity between two vectors
    """
    dot_product = 0
    length = min(len(vector1), len(vector2))

    for i in range(length):
        dot_product += vector1[i] * vector2[i]

    magnitude1 = math.sqrt(sum(x * x for x in vector1))
    magnitude2 = math.sqrt(sum(x * x for x in vector2))

    return dot_product / (magnitude1 * magnitude2)

def get_image_embedding(image, endpoint, key, max_attempts=20):
    """
    This function generates image embedding using Azure Computer Vision 4.0.
    It will try up to max_attempts times, and if it is successful, it will return 
    the image embedding. If it is unsuccessful, it will return None.
    """
    num_attempts = 0

    while num_attempts < max_attempts:
        img_embedding = image_embedding(image, endpoint, key)
        if img_embedding is None:
            num_attempts += 1
        else:
            return img_embedding
        
    return None

def get_image_embedding_multiprocessing(images, endpoint, key, max_attempts=20, max_workers=4):
    """
    This function generates embeddings for a collection of images using multiprocessing.
    Reference: https://github.com/retkowsky/Azure-Computer-Vision-in-a-day-workshop
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(
            tqdm(
                executor.map(lambda x: get_image_embedding(x, endpoint, key, max_attempts), images),
                total=len(images),
            )
        )

    return [emb for emb in embeddings if emb is not None]

def similarity_results(image_files, image_embeddings, reference_emb):
    """
    Returns cosine similarity result between an image collection and the 
    search vector (image or text embeddings)
    """
    similarity_values = [
        get_cosine_similarity(image_emb, reference_emb)
        for image_emb in image_embeddings
    ]

    df_files = pd.DataFrame(image_files, columns=['image_file'])
    df_similarity = pd.DataFrame(similarity_values, columns=['similarity'])
    df = pd.concat([df_files, df_similarity], axis=1)
    df.sort_values('similarity',
                   axis=0,
                   ascending=False,
                   inplace=True,
                   na_position='last')
    return df

def get_topn_images(df, topn=5):
    """
    Returns the topN similar images and the corresponding similarity values
    """
    img_list = [df.iloc[i]['image_file'] for i in range(topn)]
    similarity_list = [df.iloc[i]['similarity'] for i in range(topn)]

    return img_list, similarity_list

def display_image_grid(image_list, captions, title, nrows, ncols):
    
    num_images = len(image_list)
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(Image.open(image_list[i]))
            ax.set_title(captions[i])
            ax.axis('off')
        else:
            ax.axis('off')
    f.suptitle(title, fontsize=16)
    f.subplots_adjust(top=0.9)
    plt.show()

def search_by_image(reference_image, image_files, image_embeddings, 
                    endpoint, key, topn=5, disp=True):
    """
    Search for similar images using an image as a reference
    """
    
    # Find the embedding of the reference image
    ref_emb = image_embedding(reference_image, endpoint, key)
    # Similarity results
    df = similarity_results(image_files, image_embeddings, ref_emb)
    # Get topN similar images
    images, similarity_values = get_topn_images(df, topn)
    # Display similar images in a grid
    if disp:
        images.insert(0, reference_image)
        captions = [f"Top {i+1}: {os.path.basename(images[i+1])}\n"
                    f"Similarity = {round(similarity_values[i], 4)}" 
                    for i in range(topn)]
        captions.insert(0, "Reference Image")
        ncols = 3
        nrows = math.ceil(len(images)/ncols)
        display_image_grid(images, captions, 'Search results', nrows, ncols)
    return df

def search_by_text(prompt, image_files, image_embeddings, 
                   endpoint, key, topn=5, disp=True):
    """
    Search for similar images using a text prompt
    """
    
    # Find the embedding of the text prompt
    ref_emb = text_embedding(prompt, endpoint, key)
    # Similarity results
    df = similarity_results(image_files, image_embeddings, ref_emb)
    # Get topN similar images
    images, similarity_values = get_topn_images(df, topn)
    # Display similar images in a grid
    if disp:
        captions = [f"Top {i+1}: {os.path.basename(images[i])}\n"
                    f"Similarity = {round(similarity_values[i], 4)}" 
                    for i in range(topn)]
        ncols = 3
        nrows = math.ceil(len(images)/ncols)
        display_image_grid(images, captions, 'Search results for: "' + prompt + '"', nrows, ncols)
    return df