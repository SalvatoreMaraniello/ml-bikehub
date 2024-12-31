# UrbanBike: forecasting bike trips duration

In this repo, we build a ML model that predicts the duration of the bike trips from the time a bike is picked up from a docking station to the time it is returned to a docking station. This model is intended to be part of a larger service aiming at forecasting available bicycle capacity over the next 30–90 mins for app customers.

- The ML model is built starting from data relating to their bike docking stations, historical trips, and weather data (see [data](data) folder).

- The model development is showcased in the notebooks in the [studies](studies) folder. See the [Repo content](#repo-content) section for more details.

- The model is served as a REST API (see [Development setup](#development-setup) for testing the REST API locally).


## Repo content

- [studies](studies): includes the following notebooks (numbered in reading order):

  1. [00-dataset-overview.ipynb](studies/00-dataset-overview.ipynb). A quick overview of dataset provided. Data are not merged into a single dataset for analysis. *You can skip this if in a hurry*.
  2. [01-data-prep.ipynb](studies/01-data-prep.ipynb). Data from different sources is merged together into a unique dataset for exploratory data analysis and training. 
  3. [02-exploratory-data-analysis.ipynb](studies/02-exploratory-data-analysis.ipynb). Exploratory data analysis, complementing [00-dataset-overview.ipynb](studies/00-dataset-overview.ipynb).
  4. [03-model-training.ipynb](studies/03-model-training.ipynb) This notebook includes model training and selection.

- [src](src): includes a few libraries used by the notebooks and scripts in [studies](studies) and [scripts](scripts)

- [app](app): source code for a minimal REST API. See [Rest API testing](#rest-api-testing) for local testing from terminal and/or using Docker.

<!-- 
- Testing tools are found [./test](`test`). See [Testing](#Testing) section.
 -->


# Development setup

## Prerequisites

In your development machine, build a Python environment:

```sh
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

<!-- 
## Testing
From project root run:

```sh 
python -m pytest
```

- Note that tests in `test/test_api_requests.py` will fail if API is not available (see [REST API testing](#REST-API-testing)).

- To test only API endpoints:
  ```sh
  python -m pytest test/test_api_requests.py
  ``` -->


# REST API testing

Follow these instructions to test the REST API. 


## Start server

You can test the `uvicorn` server locally both from terminal or using Docker:

- From terminal:

  - Activate environment `source venv/bin/activate`

  - run:

    ```sh
    uvicorn app.main:app --host localhost --port 5000
    ```

    where you can change _host_ and _port_ as you need.

- Using Docker:

  - Build the image

    ```sh
    docker build -t bike-trip-time-model .
    ```

  - Create/Run a container using the snippet below (or from *Docker desktop* if you have it installed): 
    ```sh
    docker run -p 0.0.0.0:5000:5000  \
        --name bike-trip-time-model-container \
        bike-trip-time-model \
        uvicorn app.main:app --host 0.0.0.0 --port 5000      
    ```

- check is all ok: 
  ```sh
  python -m pytest test/test_api_requests.py
  ```


## Send requests to the API

The service is available at `http://localhost:5000`.

See [`http://localhost:5000/docs`](http://localhost:5000/docs) for a list of available endpoints - from where you can also send requests to the server.

You can also send requests from terminal using `curl`:
```sh
curl -X 'POST' \
  'http://localhost:5000/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "avg_duration_prev_7days": 13.5,
  "HPCP": 0.2,
  "dow": 5,
  "is_registered": false,
  "has_trace": false
}'
```

  <!-- - using python (see [`test/test_api_requests.py`](test/test_api_requests.py) for more):
    ```python 
    # run from project root or update `path_to_file`
    import requests

    path_to_file = 'test/payload.json'

    with open(path_to_file, 'rb') as fp:
        response = requests.post(
            'http://localhost:5000/api/v1/predict'
        )

    if response.status_code == 200:
        entities = response.json()
    else:
        raise requests.exceptions.RequestException(
            f'Something went wrong. Got status {response.status_code}.'
        )
    ``` -->
