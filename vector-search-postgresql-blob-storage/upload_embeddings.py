import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

# Azure CosmosDB for PostgreSQL
postgres_host = os.getenv("POSTGRES_HOST")
postgres_database_name = os.getenv("POSTGRES_DB_NAME")
postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
sslmode = "require"
table_name = os.getenv("POSTGRES_TABLE_NAME")
postgres_connection_string = (
    f"host={postgres_host} user={postgres_user} dbname={postgres_database_name} "
    f"password={postgres_password} sslmode={sslmode}"
)

# Data file
full_path = os.path.realpath(__file__)
working_directory = os.path.dirname(full_path)
embeddings_folder = "embeddings"
data_filename = "data.csv"
data_filepath = os.path.join(working_directory, embeddings_folder, data_filename)

postgresql_pool = psycopg2.pool.SimpleConnectionPool(1, 20, postgres_connection_string)
if (postgresql_pool):
    print("Connection pool created successfully")

# Get a connection from the connection pool
conn = postgresql_pool.getconn()
cursor = conn.cursor()

print("Saving data to table...")

# Create a temporary table with the same structure
cursor.execute("CREATE TEMPORARY TABLE tmp (filename TEXT PRIMARY KEY, embedding VECTOR(1024)) ON COMMIT DROP;")

# Copy the data from the csv file to the temporary table
with open(data_filepath) as csv_file:
    cursor.copy_expert("COPY tmp FROM STDIN CSV", csv_file)

# Insert the data from the temporary table into the original table
# taking into account any conflicts that may arise due to duplicate keys
cursor.execute(
    f"""INSERT INTO {table_name} (filename, embedding)
        SELECT * FROM tmp
        ON conflict (filename) DO UPDATE
        SET embedding = EXCLUDED.embedding;"""
)
conn.commit()

# Close the connection
cursor.close()
conn.close()
print("Done!")