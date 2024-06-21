# Use the official TensorFlow runtime as a parent image

FROM diyor28/tensorflow:2.7.3-py37
#RUN python -c "import tensorflow; print(tensorflow.__version__)"
#FROM --platform=linux/arm64/v8 tensorflow/tensorflow:2.10.00-arm64

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./generate_embeddings.py /app
COPY ./requirements.txt /app
COPY ./glove.6B.100d.txt /app

# Install any additional needed packages specified in requirements.txt
# Note: TensorFlow images already include h5py and other common packages
RUN pip install --no-cache-dir -r requirements.txt

# Create a data directory
RUN mkdir /app/data

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run embedding_app.py when the container launches
CMD ["uvicorn", "generate_embeddings:embedding_app", "--host", "0.0.0.0", "--port", "8000"]