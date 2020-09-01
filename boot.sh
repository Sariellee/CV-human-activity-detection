docker build -t eora-test .
# specify STREAM_URL or YOUTUBE_URL if you want
docker run -e STREAM_URL= -p 80:80 eora-test