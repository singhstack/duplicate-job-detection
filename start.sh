docker-compose up --build -d

# echo sleep 55 secs
# echo 20 seconds
# echo 15 secs
# done waiting
sleep 60
cd ./data


curl -X POST "http://localhost:3000/process_csv" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@jobpostings_test.csv"
sleep 10
curl -X POST "http://localhost:3000/search_csv" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@jobpostings_test.csv"

#curl -X POST "http://localhost:8001/search" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@jobpostings_test_preprocessed_embeddings.parquet"