import io
import os
from typing import List, Union
import requests
import csv
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

# Azure AI Vision credentials
vision_endpoint = os.getenv("CV_ENDPOINT") + "computervision/"
vision_key = os.getenv("CV_KEY")
version = "?api-version=2023-02-01-preview&modelVersion=latest"
vectorize_img_url = vision_endpoint + "retrieval:vectorizeImage" + version

# Azure Blob Storage credentials
blob_account_name = os.getenv("BLOB_ACCOUNT_NAME")
blob_account_key = os.getenv("BLOB_ACCOUNT_KEY")
blob_endpoint_suffix = os.getenv("BLOB_ENDPOINT_SUFFIX")
blob_connection_string = (
    f"DefaultEndpointsProtocol=https;AccountName={blob_account_name};"
    f"AccountKey={blob_account_key};EndpointSuffix={blob_endpoint_suffix}"
)
container_name = os.getenv("CONTAINER_CLIENT")
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
container_client = blob_service_client.get_container_client(container=container_name)

# Directory for saving the embeddings
full_path = os.path.realpath(__file__)
working_directory = os.path.dirname(full_path)
embeddings_folder = "embeddings"
data_filename = "data.csv"
data_filepath = os.path.join(working_directory, embeddings_folder, data_filename)

# Number of images in a batch
batch_size = 1000

def compute_image_embedding_from_blob(
        image_file: str,
) -> Union[List[float], None]:
    """
    Computes vector embeddings for an image stored in Azure Blob Storage using 
    Azure Computer Vision 4.0 (Vectorize Image API).

    :param image_file: str: the filename of the image

    :return: The vector embeddings or None if an error has occurred
    """
    headers = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": vision_key
    }
    
    try:
        # Download image to a stream
        stream = io.BytesIO()
        blob_client = container_client.get_blob_client(image_file)
        blob_client.download_blob().readinto(stream)
        stream.seek(0)

        r = requests.post(vectorize_img_url, data=stream, headers=headers)
        if r.status_code == 200:
            image_vector = r.json()["vector"]
            return image_vector
        else:
            print(
                f"An error occurred while making request for {image_file}. "
                f"Error code: {r.status_code}. Error message: {r.reason}."
            )
    except Exception as e:
        print(f"An error occurred while processing {image_file}: {e}")

    return None

def compute_image_embeddings_multithreading(
        image_files: List[str],
        max_workers: int = 4,
) -> List[Union[List[float], None]]:
    """
    Generates vector embeddings for a collection of images using multithreading.

    :param image_files: List[str]: the filenames of the images
    :param max_workers: int: the number of threads

    :return: A list containing the vector embeddings for each image in the collection
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(
            tqdm(
                executor.map(compute_image_embedding_from_blob, image_files),
                total=len(image_files),
            )
        )

    return embeddings

def save_data_to_csv(data: List[List[str]]) -> None:
    """
    Appends a list of image filenames and their corresponding embeddings to a CSV file.

    :param data: List[List[str]]: the data to append to the CSV file
    """
    with open(data_filepath, 'a', newline='') as csv_file:
        write = csv.writer(csv_file)
        write.writerows(data)

if __name__ == "__main__":
    # Set-up folder and output file
    os.makedirs(embeddings_folder, exist_ok=True)
    if os.path.exists(data_filepath):
        os.remove(data_filepath)
    
    # Get the URLs of images stored in the Blob Storage
    image_names = list(container_client.list_blob_names())
    number_of_images = len(image_names)
    print(f"Number of images in the container: {number_of_images}")

    image_names_batches = [image_names[i:(i + batch_size)] for i in range(0, number_of_images, batch_size)]
    batch_counter = 1
    for image_files in image_names_batches:
        print(f"Processing Batch {batch_counter}:")
        embeddings = compute_image_embeddings_multithreading(image_files)
        valid_data = [
            [image_files[i], str(embeddings[i])] for i in range(len(image_files))
            if embeddings[i] is not None
        ]
        print(f"Number of valid embeddings: {len(valid_data)}")
        # Save data to csv
        print(f"Saving embeddings to CSV file '{data_filename}'")
        save_data_to_csv(valid_data)
        print(f"Batch {batch_counter} - Done!")
        batch_counter += 1

    print("Done!")