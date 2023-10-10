import os
import csv
from typing import List
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

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

# Paintings dataset
full_path = os.path.realpath(__file__)
working_directory = os.path.dirname(full_path)
dataset_filename = "painting_dataset_2021.csv"
paintings_dataset = os.path.join(working_directory, dataset_filename)

def upload_image_from_url(image_url: str) -> None:
    """
    Uploads the image from a given URL to Azure Blob Storage.

    :param image_url: str: the URL of the image to upload
    """
    blob_name = image_url.split("/")[-1]
    try:
        blob_client = container_client.get_blob_client(blob=blob_name)
        blob_client.upload_blob_from_url(source_url=image_url, overwrite=True)
    except Exception as e:
        print(f"Couldn't upload image {blob_name} to Azure Storage Account due to error: {e}")

def upload_images(
        images_url: List[str],
        max_workers: int = 4,
) -> None:
    """
    Uploads a collection of images to Azure Blob Storage. Each image is specified by a URL.

    :param image_url: List[str]: the URLs of the images to upload
    :param max_workers: int: the maximum number of threads to use
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(upload_image_from_url, images_url),
                total=len(images_url),
            )
        )

if __name__ == "__main__":
    # Find the URLs of the images in the dataset
    with open(paintings_dataset, mode='r') as dataset_csv:
        csv_reader = csv.DictReader(dataset_csv, skipinitialspace=True)
        images_url = [row["Image URL"] for row in csv_reader]
    valid_images_url = [image_url for image_url in images_url if image_url != "N/A"]
    unique_images_url = list(set(valid_images_url))

    print(f"Number of images in the dataset: {len(unique_images_url)}")
    print(f"Uploading images to container '{container_name}'")

    # Upload images to blob storage
    upload_images(valid_images_url)

    print("Done!")