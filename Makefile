IMAGE_NAME := car-app
DOCKER_REPO := ifeademola
IMAGE_VERSION := 0.0.1

build:
    docker build -t "$(DOCKER_REPO)/$(IMAGE_NAME):$(IMAGE_VERSION)" .

push:
    docker push $(DOCKER_REPO)/$(IMAGE_NAME):$(IMAGE_VERSION)

predict:
	python inference.py --model_path artifacts/models/model.joblib \
                    --ct_path artifacts/models/ct.joblib \
                    --fueltype gas \
                    --horsepower 4500


