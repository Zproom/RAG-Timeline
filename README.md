# Instructions for setting up the persistent Qdrant repository

Qdrant by default executes in memory which is not ideal as we'd need to pull the data everytime we restart the application / qdrant.

Therefore, we will mount a local volume to act as the repository. See below for the instructions:

1) Ensure docker is installed.
2) create a directory in home directory: (example below names directory -> qdrant)

mkdir -p ~/qdrant_storage

3) run the docker instance via the following command. Then when we connect with the qdrant-client python library ensure the ports match (use 6333 by default)

docker run -d --name qdrant -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant



