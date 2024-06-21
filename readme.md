# Job Duplicates Detection System with Milvus and Docker

## Overview

This project demonstrates a scalable approach to detect duplicate job postings using Glove embeddings and Milvus, a high-performance vector search engine. The system is containerized using Docker for easy deployment and management.

### Directory Structure

jobduplicates/
│
├── data/                   <- Placeholder for input data CSV file
├── data_ingestion/         <- Service for data ingestion and preprocessing
├── embeddings/             <- Service for embedding generation and duplicate detection
├── volumes/                <- Directory for persistent storage (remove before bundling)
├── embedEtcd.yaml          <- Configuration file for etcd service
├── cleanup.sh              <- Script to clean up environment
├── start.sh                <- Script to start Docker services
└── docker-compose.yml      <- Docker Compose file to orchestrate services


## Project Structure and Workflow

### Part 1: Generating Embeddings

1. **Data Ingestion and Preprocessing (data-ingestion service):**
   - Reads job postings data from `data/`, cleans and preprocesses it.
   - Sends preprocessed data to the embedding generation service.

2. **Embedding Generation (embeddings service):**
   - Sets up Milvus for vector storage and search.
   - Generates embeddings using `<YOUR_MODEL>` (e.g., Sentence Transformers, Word2Vec).
   - Inserts embeddings into Milvus collections.
   - Detects duplicate job postings based on cosine similarity.
   - The threshold was identified in analysis folder and is detailed later in this section.

### Part 2: Setting Up Milvus for Duplicate Detection

- **Milvus Standalone:** Provides backend for vector storage and efficient similarity search.
- **etcd:** Configures and manages Milvus instances.
- **Minio:** Handles data storage for Milvus.

### Definition of Duplicate Jobs

Duplicate jobs are defined as postings sharing identical attributes in key fields:
- Company
- Location
- Title
- Description
- Skills
- Same Status (Both Open or Both Closed)

## Instructions for Docker Setup and Execution

### Prerequisites

- Docker and Docker Compose installed on your machine.
- Place your input data CSV file (`<YOUR_DATASET>.csv`) in the `data/` directory.
- Place your GloVe embedding text file (`glove.6B.100d.txt`) in the `embeddings/` directory.



### Steps to Run

1. **Clone Repository:**
   ```bash
   git clone <repository_url>
   cd jobduplicates

2. **Run Docker compose using start.sh:**
   bash
   bash start.sh

This will build your docker, run it and generate duplicate job pairs for you in /embeddings as 'duplicate job pairs.csv'


### Identifying Optimal threshold

### Similarity Matrix Generation

- A dense \( n \times n \) similarity matrix was generated using embedding vectors of job postings.
- Each element in the matrix represents the similarity score between pairs of job postings.

### Key Generation for Validation

- Job postings were uniquely identified using hashed job descriptions combined with selected features such as job title, company name, location, and skills.
- These keys served as ground truth labels to validate duplicate pairs.

### Threshold Evaluation

- From the similarity matrix, pairs of job postings and their similarity scores were extracted.
- Different threshold values (from 0 to 1 in increments of 0.05) were tested to classify pairs with similarity scores above the threshold as duplicates.

### Metrics Used

- **Precision:** Measures the accuracy of identifying duplicate pairs among all predicted duplicates.
- **Recall:** Measures the ability to correctly identify duplicate pairs among all actual duplicates.
- **F1-score:** Harmonic mean of precision and recall, providing a balanced assessment of duplicate detection performance.

### Optimal Threshold Selection

- The threshold that maximized the F1-score was chosen as the optimal threshold for duplicate detection.
- F1-score was selected due to its ability to balance precision and recall, ensuring effective identification of duplicates while minimizing false positives.

By systematically evaluating these metrics across various threshold values, we established an efficient approach to detect duplicate job postings using Milvus.



## Responses to Questions:

### Part 1

Question 2: Justify your choice: Explain why you chose this method and its potential advantages for this task and its potential advantages in the context of duplicate detection.

Response: I opted to use pretrained GloVe embeddings (`glove.6B.100d.txt`) for generating embeddings for the job descriptions due to their efficiency and effectiveness in capturing semantic relationships between words. GloVe embeddings are pre-trained on large corpora, ensuring they have a rich understanding of various contexts and nuances in the language. By generating embeddings for fields like job description, title, company, location, and skills, and then concatenating them, we create a comprehensive representation of each job post.

This method leverages GloVe's ability to capture word similarities and contextual information, which is crucial for accurately detecting duplicates in the job postings data. Additionally, GloVe embeddings are computationally efficient and can be easily integrated with TensorFlow, ensuring a streamlined and reproducible workflow.

This can be worked upon to improve the accuracy of the system by choosing a finetuned glove embedding, domain specific embeddings, ensemble of embeddings, contextual embeddings using BERT, RoBERTa or other tranformer based embedding vectors.

### Part 2

Question 3 b: Explain any required environment variables or configurations.
Response: M1 Macbook. Using specific tensorflow version due to protobuf compatibility issues between tensorflow and pymilvus. Refer to Dockerfile in /embeddings.

Question 4: Evaluate the results. How effective is the method in detecting duplicates?
Response: The threshold of 0.95 similarity gave the us a precision of 0.000120, recall of 0.133333, and f1 score of 0.000240. The peformance of this method can be improved in a number of ways given proper time:
- Improving the quality of the embedding vectors, as discussed in previous response, Looking at different approaches such as weighted average of embedding of individual features, early fusion and late fusion while estimating similarity.
- Better feature representation of a job posting including considering other features such as qualification, better preprocessing of text
- Using a larger sample size than 1000 to identify a more optimal threshold

Question 5: How did you decide on the threshold for determining duplicates in Milvus? Which metrics are you using?
Response: Threshold Determination for Duplicates in Milvus
##### Methodology:
1. **Similarity Matrix**: Generated a similarity matrix for a subset of job postings using embedding vectors.
2. **Key Generation**: Created unique keys for job postings by hashing job descriptions and concatenating selected features.
3. **Threshold Evaluation**: Evaluated potential thresholds (0 to 1) by classifying pairs with similarity scores above the threshold as duplicates.
4. **Metrics**:
    - **Precision**: Correctly identified duplicates / Total predicted duplicates.
    - **Recall**: Correctly identified duplicates / Total actual duplicates.
    - **F1-score**: Harmonic mean of precision and recall.

##### Optimal Threshold:
Using a smaller subset of the dataset, the threshold that maximized the F1-score was selected. This approach balances precision and recall, ensuring effective duplicate detection. The optimal threshold is then applied to the larger dataset for efficient duplicate identification.

##### Other Approaches:
- **Receiver Operating Characteristic (ROC) Curve**: Plotting true positive rate vs. false positive rate and selecting the threshold with the highest area under the curve (AUC).
- **Precision-Recall Curve**: Plotting precision vs. recall for different thresholds and choosing the point that maximizes the F1-score or another relevant metric.
- **Grid Search with Cross-Validation**: Splitting the dataset into training and validation sets to iteratively find the best threshold.
- **Clustering**: Using clustering algorithms like DBSCAN to identify natural groupings of duplicates based on similarity scores.
- **Silhouette Analysis**: Evaluating the consistency within clusters of duplicates and non-duplicates to select an optimal threshold.


Question 6: Can you think of other use-cases for the embeddings you generated, beyond duplicate detection?
Response: Generated embeddings can enhance job platforms beyond duplicate detection by enabling personalized job recommendations, semantic search, automated resume matching, job classification, and market segmentation. They can optimize job descriptions, extract key skills, and identify trends in skill demand. Additionally, embeddings aid in detecting fraudulent postings, ensuring diversity and inclusion in job descriptions, and improving internal mobility and referral systems. These embeddings also support generating concise job summaries and highlighting key content, significantly boosting the efficiency and effectiveness of HR processes.


### Part 3

Question 4: How would you handle real-time data streaming into this system?
Response: ## Handling Real-Time Data Streaming for Duplicate Job Detection

### Approach Overview

In designing a system for real-time duplicate job detection, several critical considerations come into play to ensure efficiency, scalability, and accuracy. Here’s a structured approach to address the challenge:

##### 1. Data Ingestion Pipeline

- **Stream Processing Framework:** Adopt a robust stream processing framework like Apache Kafka, Apache Flink, or AWS Kinesis. This framework will manage the continuous flow of incoming job postings from various sources, such as job boards or APIs.

##### 2. Preprocessing and Cleaning

- **Real-Time Preprocessing:** Implement preprocessing steps within the stream processing framework. Tasks include noise removal, tokenization, handling special characters, and possibly stemming or lemmatization of text data.
- **Feature Extraction:** Extract key features from job postings in real-time, such as job title, description, company name, location, and skills.

##### 3. Embedding Generation

- **Integration with Embedding Service:** Integrate with an embedding generation service capable of generating embeddings for job postings in real-time.
- **Efficient Embedding Models:** Utilize efficient embedding models, such as pre-trained GloVe embeddings. Consideration should also be given to updating embeddings dynamically based on continuous learning or fine-tuning processes.

##### 4. Duplicate Detection

- **Utilization of Vector Search Engine:** Employ a vector search engine like Milvus for efficient similarity search operations in real-time.
- **Duplicate Detection Logic:** Implement duplicate detection logic using similarity thresholds derived from embeddings and feature-based keys generated during preprocessing.

##### 5. Scalability and Performance

- **Distributed Processing:** Ensure scalability by deploying components across distributed environments.
- **Load Balancing:** Use load balancers to evenly distribute incoming data streams across processing nodes, optimizing resource utilization.
- **Performance Monitoring:** Implement robust monitoring and logging mechanisms to track processing times, throughput, and system health metrics.

##### 6. Deployment Considerations

- **Containerization:** Dockerize components (stream processing, embedding generation, Milvus) for seamless deployment and scalability using Docker Compose or Kubernetes orchestration.
- **Cloud Integration:** Leverage cloud services (AWS, Google Cloud, Azure) for elastic scaling of infrastructure based on workload demands.

##### 7. Continuous Improvement

- **Feedback Mechanisms:** Implement feedback loops to continuously evaluate and enhance the duplicate detection model based on real-time performance metrics and user feedback.
- **Iterative Development:** Incorporate iterative development practices to refine embedding models and fine-tune algorithms based on evolving data patterns and job posting characteristics.

### Conclusion

Designing and implementing a real-time data streaming solution for duplicate job detection involves integrating stream processing, embedding generation, and similarity search components effectively. By leveraging scalable technologies, monitoring system performance, and iterating based on real-time insights, the system can achieve accurate and efficient duplicate detection in a dynamic operational environment.
