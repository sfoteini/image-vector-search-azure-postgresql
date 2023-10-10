# Image Vector Similarity Search with Azure Computer Vision and PostgreSQL

<p>
  <a href="https://sfoteini.github.io/blog/image-vector-similarity-search-with-azure-computer-vision-and-postgresql/" target="_blank"><img src="https://img.shields.io/badge/Instructions-informational?style=for-the-badge" alt="Tutorial"></a>
</p>

In these Jupyter Notebooks, you will explore the Image Retrieval functionality of Azure AI Vision, which is powered by the Florence Foundation model. You will:

* Use the Image Retrieval APIs in Python to search a collection of images to find those that are most similar to a given query image or text prompt.
* Use the Azure Cosmos DB for PostgreSQL to store and query vector data.

Before starting to build your image vector similarity system, follow these steps:

* Sign up for either an [Azure free account](https://azure.microsoft.com/free/?WT.mc_id=AI-MVP-5004971) or an [Azure for Students account](https://azure.microsoft.com/free/students/?WT.mc_id=AI-MVP-5004971). If you already have an active subscription, you can use it.
* Create a Cognitive Services resource in the Azure portal.
    
    > The Image Retrieval APIs are available in the following regions: East US, France Central, Korea Central, North Europe, Southeast Asia, West Europe, West US.

* Create an Azure Cosmos DB for PostgreSQL cluster.
* Install Python 3.x, Visual Studio Code, Jupyter Notebook and Jupyter Extension for Visual Studio Code.

You can check out my article ["Image Vector Similarity Search with Azure Computer Vision and PostgreSQL"](https://sfoteini.github.io/blog/image-vector-similarity-search-with-azure-computer-vision-and-postgresql/) if you want to learn more about this topic.


## Quickstart

In this quickstart, you will explore the Image Retrieval APIs of Azure AI Vision and the `pgvector` extension. Before running the Jupyter Notebooks, you should follow these steps:

1. Enable the `pgvector` extension on your Azure Cosmos DB for PostgreSQL cluster:

    ```sql
    SELECT CREATE_EXTENSION('vector');
    ```

2. Create a table:

    ```sql
    CREATE TABLE imagevectors(
        file TEXT PRIMARY KEY,
        embedding VECTOR(1024)
        );
    ```

3. Save the keys of your Azure AI Vision and Azure Cosmos DB resources in an `.env` file.

## Example: Vector Search with Azure Cosmos DB for PostgreSQL, Azure Blob Storage and Azure AI Vision

In this notebook, you will build a basic image vector similarity search application using the Azure AI Vision Image Retrieval APIs and Azure Cosmos DB for PostgreSQL.

Before you start:

1. Enable the `pgvector` extension on your Azure Cosmos DB for PostgreSQL cluster:

    ```sql
    SELECT CREATE_EXTENSION('vector');
    ```

2. Create a table:

    ```sql
    CREATE TABLE imagevectors(
        filename TEXT PRIMARY KEY,
        embedding VECTOR(1024)
        );
    ```

3. Create an Azure Blob Storage container to store the images.

4. Run the `upload_images.py` script to upload the images in your Blob Storage container. (Image source: [Visual Geometry Group - University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/paintings/))

5. Run the `generate_embeddings.py` script to compute the embeddings of all the images. The embeddings are saved in a `.csv` file in the *embeddings* folder.

6. Run the `upload_embeddings.py` script to populate the PostgreSQL table with data.
