# ASME - [A] [S]equential Recommendation [M]odel [E]valuator

# Getting Started

Configuration files for "Enhancing Sequential Next-Item Prediction through Modelling Non-Item Pages in Transformer-Based
Recommender Systems" can be found under non-items-paper-configs.

To train, install the dependencies as described below. You can train a model with ```train config_file``` and evaluate
with ```evaluate --config-file config_file --checkpoint-file checkpoint```. The ML-20m dataset will be downloaded automatically,
the Coveo dataset has to be downloaded manually. To (re-)generate the coveo files in the first run, set    ```perform_convert_to_csv: true``` in the config. 


## Install Locally
* Install [Poetry](https://python-poetry.org)
* Clone the repository
* Build the development virtual environment: `poetry install`
* Enter the virtual environment: `poetry shell`



## Development
The development image automatically pulls and updates the specified branch and uses poetry to execute asme.
### Build Container
```
podman build -f Dockerfile -t asme-dev:latest --target asme-dev
```
### Train a model
Here is an example how you can use the dev image to train a model on the latest version from the repository
```shell
podman run -e GIT_TOKEN=<token> -e REPO_USER=<user> -e REPO_BRANCH=master PROJECT_DIR=/project -v /path/to/project:/project:Z asme-dev:latest train /project/sample.jsonnet
```
### Environment Variables
* PREPARE_SCRIPT:
  If set the script will be executed before the framework command is run. This is useful for copying data to a faster drive
* GIT_TOKEN:
  Token used to checkout the repository
* REPO_USER:
  User used to checkout the repository
* REPO_BRANCH:
  Branch to checkout
* PROJECT_DIR:
  A writeable path within the container, that will be used to store the repository

## Release

### Build
```shell
podman build -f Dockerfile --target asme-release -t asme:latest
```
### Run
```shell
# with GPU support on linux
podman run -it -v /path/to/project:/project:Z --security-opt=no-new-privileges --cap-drop=ALL --security-opt label=type:nvidia_container_t asme:latest asme train /project/config.jsonnet

# without GPU support on linux
podman run -it -v /path/to/project:/project:Z asme:latest asme train /project/config.jsonnet
```
### Environment Variables
* PREPARE_SCRIPT: if set the script will be executed before the framework command is run. This is useful for copying data to a faster drive.

### Kubernetes
Both containers can be used as in a kubernetes cluster. Please follow the instructions below to correctly setup your namespace for the `Development` container.

#### Development
The development container does not come with a preinstalled version of the framework. Instead, the entrypoint will clone the specified branch and setup an environment using poetry. Your command will be executed inside this environment.

##### Generate Gitlab access token
Even though the password is supplied via an environment variable that can be populated by a kubernetes secret, you probably don't want to share it with the admins. Lucky for us, we can also generate a gitlab access token for that purpose.

1. Go to your Profile and select Settings->Access Tokens
2. Give it a name, e.g. `k8s`
3. Select `read_repository`
4. Save the final token, because it won't be accessible afterwards

##### Create kubernetes secret

```
kubectl -n <namespace> create secret generic gitlab-token --from-literal=token=xxx
```

