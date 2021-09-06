#!/usr/bin/env bash
export TAG="1.0.0"

BUILD_COMMAND="podman"
ALTERNATE_BUILD_COMMAND="docker"

if ! command -v $BUILD_COMMAND &> /dev/null
then
  echo "Could not find build comman: ${BUILD_COMMAND}, switching to alternate: ${ALTERNATE_BUILD_COMMAND}"
  BUILD_COMMAND=$ALTERNATE_BUILD_COMMAND
fi

if ! command -v $BUILD_COMMAND &> /dev/null
then
  echo "Error: You need to install docker or podman to build the container image."
  exit
fi

if ! command -v poetry &> /dev/null
then
  echo "Error: You need poetry installed to build the container."
  exit
fi

if [ -d "dist" ]; then
  echo "Cleaning up: dist... "
  rm -r dist
fi

echo "Building wheel..."
poetry build -f wheel

echo "Building container image..."
if [ "$BUILD_COMMAND" = "podman" ]
then
  # podman
  $BUILD_COMMAND build . --format docker -f k8s/release/Dockerfile --build-arg TAG=$TAG -t asme:${TAG} -t gitlab2.informatik.uni-wuerzburg.de:4567/dmir/dallmann/recommender/asme:${TAG}
else
  # docker
  $BUILD_COMMAND build . -f k8s/release/Dockerfile --build-arg TAG=$TAG -t asme:${TAG} -t gitlab2.informatik.uni-wuerzburg.de:4567/dmir/dallmann/recommender/asme:${TAG}
fi

echo "Cleaning up: dist..."
rm -r dist