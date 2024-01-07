include .env
export

push:
	docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}
	docker push rcland12/detection-stream:jetson-latest
	docker push rcland12/detection-stream:triton-latest
	docker push rcland12/detection-stream:nginx-latest
	docker images -q rcland12/detection-stream:jetson-latest | xargs -I{} docker tag {} rcland12/detection-stream:jetson-${LATEST_VERSION}
	docker images -q rcland12/detection-stream:triton-latest | xargs -I{} docker tag {} rcland12/detection-stream:triton-${LATEST_VERSION}
	docker images -q rcland12/detection-stream:nginx-latest | xargs -I{} docker tag {} rcland12/detection-stream:nginx-${LATEST_VERSION}