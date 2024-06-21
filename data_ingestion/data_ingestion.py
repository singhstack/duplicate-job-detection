from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import httpx

from io import StringIO
import uvicorn

app = FastAPI()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def clean_text(text, use_stemming=False, lemmatize = False):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize text
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Apply stemming or lemmatization
    if use_stemming:
        words = [stemmer.stem(word) for word in words]
    if lemmatize:
        words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a single string
    return ' '.join(words)

def clean_skills(skills):
  # Remove the brackets and split by comma
  skills_list = re.sub(r'[\[\]]', '', skills).split(',')
  # Clean each skill individually
  cleaned_skills = [clean_text(skill, use_stemming=False) for skill in skills_list]
  # Join the cleaned skills back into a single string
  return ', '.join(cleaned_skills)

#file path
def preprocess(df):
    df = df.dropna(subset=['Job Description'])
    df = df.reset_index(drop=True)
    df['Skills'].fillna('', inplace=True)
    df['City'].fillna('', inplace=True)
    df['Company Name'].fillna('',inplace = True)
    #Parse job description using beautiful soup to get text without html tags
    df['Job Description Parsed'] = df['Job Description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    #preprocess necessary fields
    # Apply the cleaning function to the title, description, and skills columns
    df['Job Title Cleaned'] = df['Job Title'].apply(lambda x: clean_text(x))
    df['Job Description Parsed Cleaned'] = df['Job Description Parsed'].apply(lambda x: clean_text(x))
    df['Skills Cleaned'] = df['Skills'].apply(lambda x: clean_skills(x))  # Use lemmatization for skills
    df['Company Name Cleaned'] = df['Company Name'].apply(lambda x: clean_text(x))
    df['Location'] = df['City'] + ', ' + df['State']
    df['Location Cleaned'] = df['Location'].apply(lambda x: clean_text(x))  
    return df

@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    print("initiated")
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))

    processed_df = preprocess(df)
    filename_to_save = file.filename.split('.csv')[0]+'_preprocessed.parquet'
    parquet_file_path = 'data/'+filename_to_save
    processed_df.to_parquet(parquet_file_path)
    # Send the processed parquet file to the embedding service
    async with httpx.AsyncClient() as client:
        with open(parquet_file_path, 'rb') as f:
            response = await client.post(
                'http://embedding-service:8000/generate_embeddings',
                files={'file': (filename_to_save, f, 'application/octet-stream')}
            )
    return {"message": "File processed successfully", "file_path": parquet_file_path, "embedding_response": response.json()}


@app.post("/search_csv")
async def search_csv(file: UploadFile = File(...)):
    print("initiated")
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))

    processed_df = preprocess(df)
    filename_to_save = file.filename.split('.csv')[0]+'_preprocessed.parquet'
    parquet_file_path = 'data/'+filename_to_save
    processed_df.to_parquet(parquet_file_path)
    # Send the processed parquet file to the embedding service
    async with httpx.AsyncClient() as client:
        with open(parquet_file_path, 'rb') as f:
            response = await client.post(
                'http://embedding-service:8000/search',
                files={'file': (filename_to_save, f, 'application/octet-stream')},
                timeout= None
            )
    return {"message": "File processed and sent to embedding service for search", "file_path": parquet_file_path, "embedding_response": response.json()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #main()

'''@app.post("/process_text")
async def process_text(text: str = Form(...)):
    cleaned_text = parse_job_description(text)
    return {"cleaned_text": cleaned_text}
'''


'''
def main():
  #read the file
  #read data
  file_name = 'data/cxx.csv'
  df = pd.read_csv(file_name)
  #preprocess data #shorten the time taken to preprocess
  pre_processed_data = preprocess(df)
  #save the file
  parquet_file_path = 'data/jobs_preprocessed.parquet'
  pre_processed_data.to_parquet(parquet_file_path)
'''

#accept the html input, data, 
#returns plain text
