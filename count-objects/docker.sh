IMAGE_NAME=count-cars-image
CONTAINER_NAME=count-cars-container

docker build --no-cache -t ${IMAGE_NAME} .
docker run -it --name ${CONTAINER_NAME} -v $PWD/data:/app -v $PWD/count_cars.py:/app ${IMAGE_NAME}