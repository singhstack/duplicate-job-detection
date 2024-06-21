import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    Index
)
import httpx

from io import BytesIO
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse


embedding_app = FastAPI()

# Milvus connection parameters
MILVUS_HOST = 'milvus-standalone'
MILVUS_PORT = '19530'
DB_NAME = 'jobs'
COLLECTION_NAME = 'job_posting_new'

# Load the GloVe model (download the appropriate file first)
glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False, no_header=True)

# Function to connect to Milvus and ensure collection exists
# Function to connect to Milvus and ensure collection exists
def connect_to_milvus():
    try:
        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

        # Check if collection already exists
        if utility.has_collection(COLLECTION_NAME):
            print("Collection already exists")
        else:
            print(f"Collection '{COLLECTION_NAME}' not found. Please ensure it is created.")
            raise Exception(f"Collection '{COLLECTION_NAME}' not found")

    except Exception as e:
        print(f"Error during Milvus operations: {e}")


# Function to convert text to GloVe embeddings using TensorFlow/Keras
def get_glove_embeddings(texts, glove_model, max_length=1000, embedding_dim=100):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in glove_model:
            embedding_matrix[i] = glove_model[word]
    
    # Define Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                  output_dim=embedding_dim,
                                  weights=[embedding_matrix],
                                  input_length=max_length,
                                  trainable=False),
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    
    # Get embeddings
    embeddings = model.predict(padded_sequences)
    return embeddings

def concatenate_embeddings(row):
  return np.concatenate([row['Job Description Parsed Cleaned Embedding'],row['Job Title Cleaned Embedding'],row['Company Name Cleaned Embedding'],row['Skills Cleaned Embedding'],row['Location Cleaned Embedding']])

def embedding_helper(df):
    # Process all texts in each column at once
    columns_to_process = ['Job Title Cleaned', 'Job Description Parsed Cleaned', 'Skills Cleaned', 'Company Name Cleaned', 'Location Cleaned']
    print(df.columns)
    print(df)
    text_data = df[columns_to_process].astype(str).values.flatten() #why is there float data there in the first place??

    # Get embeddings for text data with progress bar
    embeddings = []
    num_rows = len(df)

    print("generating embedding vectors for job postings")
    for i in tqdm(range(0, len(text_data), num_rows)):
        embeddings.extend(get_glove_embeddings(text_data[i:i+num_rows], glove_model))

    embeddings = get_glove_embeddings(text_data, glove_model)
    num_rows = len(df)
    embedding_dim = embeddings.shape[1]
    start_idx = 0
    for col in columns_to_process:
        end_idx = start_idx + num_rows
        col_embeddings = embeddings[start_idx:end_idx]
        df[f'{col} Embedding'] = list(col_embeddings)
        #df[f'{col} Embedding'] = embeddings[start_idx:end_idx]
        start_idx = end_idx

    df['concatenated_embeddings']  = df.apply(concatenate_embeddings, axis=1)

    return df

async def create_index():
    index_params = {
        "collectionName": COLLECTION_NAME,
        "indexParams": [
            {
                "metricType": "COSINE",
                "fieldName": "vector",
                "indexName": "vector_index",
                "indexConfig": {
                    "index_type": "IVF_FLAT",
                    "nlist": "128"
                }
            }
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"http://{MILVUS_HOST}:{MILVUS_PORT}/v2/vectordb/indexes/create", json=index_params)
        if response.status_code == 200:
            print("Index created successfully")
        else:
            print(f"Failed to create index: {response.text}")


@embedding_app.on_event("startup")
async def create_collection():
    try:
        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

        # Define schema for the collection
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, description="job id", max_length=535)
        embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=500, description="vector")
        # job_field = FieldSchema(name="job_raw", dtype= DataType.VARCHAR, description="job details", max_length= 535)
        schema = CollectionSchema(fields=[id_field, embedding_field], description="Job Embeddings Collection")

        # Check if collection already exists
        if utility.has_collection(COLLECTION_NAME):
            print("collection already exists")
        else:
            # Create the collection
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            print(f"Created collection '{COLLECTION_NAME}'")

            await create_index()

    except Exception as e:
        print(f"Error during collection creation: {e}")


# we want a @embedding_app.get("/search_duplicates")
# once we do that it takes as input what? dedeplicated list of job ids with counts next ot it
def compute_similarity_matrix(vectors):
    num_vectors = len(vectors)
    similarity_matrix = np.zeros((num_vectors, num_vectors))
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    for i in range(num_vectors):
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        results = collection.search(data=[vectors[i]], anns_field="vector", param=search_params, limit=num_vectors)
        
        for result in results[0]:
            j = result.id
            similarity_matrix[i][j] = result.distance
    
    return similarity_matrix


@embedding_app.post("/search")
async def search(file: UploadFile = None):
    connect_to_milvus()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    
    # if file is None:
    #     expr = ""
    #     vectors = collection.query(expr=expr, output_fields=["vector"])
    #     vectors = [v['vector'] for v in vectors]
    #     similarity_matrix = compute_similarity_matrix(vectors)
    #     return JSONResponse(content={"similarity_matrix": similarity_matrix.tolist()})
    # else:
    content = await file.read()
    df = pd.read_parquet(BytesIO(content))
    
    #if len(df) != 1:
    #    return JSONResponse(content={"error": "The CSV file must contain exactly one row."})
    df_embeddings = embedding_helper(df)
    query_vectors = df_embeddings.iloc[:]['concatenated_embeddings'].tolist()
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    threshold = 0.95 #identified in optimal_threshold.ipynb
    n = len(df_embeddings)
    similarity_matrix = np.zeros((n, n))
    job_pairs = []
    for i,vector in enumerate(query_vectors):
        search_results = collection.search(
            data=[vector],
            anns_field="vector", param=search_params, limit=n)

        for result in search_results[0]:
            job_id = result.id  # Get the job ID from the search result
            j = df_embeddings.index[df_embeddings['Job Id'] == job_id].tolist()  # Find the row number
        
            if j:  # Ensure that j is not empty
                j = j[0]  # Get the first match (assuming unique job IDs)
                similarity_matrix[i, int(j)] = result.distance 
                if result.distance > threshold:  # Check if the similarity score is over the threshold
                    if df_embeddings['Status'].iloc[i] == df_embeddings['Status'].iloc[j]:
                        job_pairs.append((df_embeddings['Job Id'].iloc[i], job_id, result.distance, True))

    # Create a DataFrame from the collected job pairs
    job_pairs_df = pd.DataFrame(job_pairs, columns=['job_id_1', 'job_id_2', 'similarity', 'is_duplicate'])
    job_pairs_df.to_csv('duplicate job pairs.csv', index=False)
    #job_ids = [result.id for result in results[0]]
    #return {"job_ids": job_ids}
    #print(f'similarity matrix  shape: {similarity_matrix.shape}')
    # Convert numpy array to pandas DataFrame
    #df_similarity_matrix = pd.DataFrame(similarity_matrix)

    # Save DataFrame as a CSV file
    #df_similarity_matrix.to_csv('similarity_matrix_test.csv', index=False)
    #df_embeddings.to_csv('job_postings_test_embeddings.csv')
    print(f'{len(job_pairs_df)} duplicate jobs identified')
    return {'Number of Duplicate job pairs identified shape': job_pairs_df.shape[0]}

@embedding_app.post("/generate_embeddings")
async def generate_embeddings(file: UploadFile = File(...)):
    try:
        print("embedding vector generation initiated")
        content = await file.read()
        df = pd.read_parquet(BytesIO(content))

        df_embeddings = embedding_helper(df)
        filename_to_save = file.filename.split('.parquet')[0]+'_embeddings.parquet'
        parquet_file_path = 'data/'+filename_to_save
        df_embeddings.to_parquet(parquet_file_path)
        
        # Connect to Milvus
        connect_to_milvus()
        #try:
        collection = Collection(name=COLLECTION_NAME)

        # Prepare data for insertion into Milvus collection
        ids = [str(row['Job Id']) for _, row in df_embeddings.iterrows()]
        vectors = [row['concatenated_embeddings'].tolist() for _, row in df_embeddings.iterrows()]
        #job_raw = [str(row.to_dict()) for _, row in df_embeddings.iterrows()]
            
        data = [ids, vectors]
        print(f"LOG : Entered {len(ids)} rows")
        # print(data)

        # Insert data into Milvus collection
        mr = collection.insert(data)
        ids = mr.primary_keys

        if ids:
            print(f"Successfully inserted {len(ids)} embeddings into collection '{COLLECTION_NAME}'")

            # Prepare response
            return {"message": "Embeddings inserted successfully into Milvus", "num_embeddings": len(ids)}
        else:
            print(f"Failed to insert embeddings into collection '{COLLECTION_NAME}'")

    except Exception as e:
        print(f"Error during embeddings insertion into Milvus: {e}")

    return {"message": "Failed to insert embeddings into Milvus"}

if __name__ == "__main__":
    uvicorn.run(embedding_app, host="0.0.0.0", port=8000)