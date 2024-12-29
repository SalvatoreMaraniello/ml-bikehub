# UrbanBike: bicycle availability forecasting

This model forecasts available bicycle capacity over the next 30–90 mins for app customers.

A key step in predicting capacity is to predict the duration of the bike trips from the time a bike is picked up from a docking station to the time it is returned to a docking station. To make these predictions, UrbanBike has given us data relating to their bike docking stations, historical trips, and weather data.

The model is served as a REST API (see [Development setup](#development-setup) for testing the REST API locally).


# Baseline model
- Average time and weather.


## Repo content

- `src`: 

- `app`: source code for minimal REST API.


- Testing tools are found [./test](`test`). See [Testing](#Testing) section.



# Development setup

## Prerequisites

In your development machine, build a Python environment:

```sh
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Testing
From project root run:

```sh 
python -m pytest
```

- Note that tests in `test/test_api_requests.py` will fail if API is not available (see [REST API testing](#REST-API-testing)).

- To test only API endpoints:
  ```sh
  python -m pytest test/test_api_requests.py
  ```



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

- For available endpoints, see [`http://localhost:5000/docs`](http://localhost:5000/docs), from where you can also send requests to the server!

- To send requests programmatically:

  - from shell using `curl` (see [`making-api-requests.md`](docs/making-api-requests.md) for more):
    ```sh
    # Requires run from project root, else update path to PDF
    curl -X 'POST' \
      'http://localhost:5000/api/v1/predict' \
      -H 'Content-Type: multipart/form-data'
    ```

  - using python (see [`test/test_api_requests.py`](test/test_api_requests.py) for more):
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
    ```
