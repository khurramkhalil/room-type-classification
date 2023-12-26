# Pull a base image.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         nginx \
         git \
         vim \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /opt/program

# Install requirements
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY . .
# download weights
RUN gdown --id 1Qntn0Z4hAp_6XvA8g7nhHXYvrm_AulFi --output trained_weights/room-type-classification-clip-v1.0.0.pt


EXPOSE 8080
CMD ["uvicorn", "app:APP", "--host", "0.0.0.0", "--port", "8080"]