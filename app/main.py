import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from fastapi import FastAPI

from .models import TestModel, OutputModel, InputModel
# from . import exceptions

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: agree standard for versioning
# TODO: port number hardcoded. Is this best solutions for production server?
VERSION_MAJOR, VERSION_MINOR = '1', '0.0'
app = FastAPI(
    title='Bike Trip Duration Prediction API',
    description=(
        'This API forecasts the bike trip duration based on data available prior the trip start.'),
    version=VERSION_MINOR + '.' + VERSION_MINOR,
    servers=[{'url': 'http://localhost:5000', 'description': 'Development server'}],
    debug=True,)
API_PATHS_ROOT = f'/api/v{VERSION_MAJOR}/'
API_PATHS_TESTING = API_PATHS_ROOT + 'test/'


# Load model
model = joblib.load(Path(__file__).resolve().parent / "trip-duration-predictor.joblib")


@app.get('/')
async def root():
    return {
        'message': 'Hello from Bike Trip Duration Prediction API! Find documentation at `docs` endpoint.'
    }


# Some GCP services require health endpoint.
@app.get('/health', status_code=200)
def health():
    return {}


@app.post(API_PATHS_ROOT+'predict')
async def predict(payload: InputModel) -> OutputModel:
    '''Endpoint to predict bike trip duration.'''

    # convert into a python dict ...
    payload_dict = payload.model_dump(mode='python')

    # process payload (cast into a pandas dataframe)
    dow = payload_dict.pop('dow')

    df_payload = pd.DataFrame([payload_dict])
    df_payload['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df_payload['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    df_payload['is_registered'] = df_payload['is_registered'].astype(int)
    df_payload['has_trace'] = df_payload['has_trace'].astype(int)

    # predict
    pred = model.predict(df_payload)[0]
    output = OutputModel(
        duration=float(pred)
    )
    return output


@app.post(API_PATHS_TESTING+'send')
async def dummy_send(payload: TestModel) -> TestModel:
    '''A test endpoint that returns the request body. Useful to ensure we can send a payload as 
    intended.'''
    return payload


@app.post(API_PATHS_TESTING+'send_process')
async def dummy_send_process(payload: TestModel) -> TestModel:
    '''A test endpoint that returns the request body upon manipulation. Useful to ensure the we can
    send the payload and transform it into a python object as intended.'''

    # convert into a python dict ...
    params = payload.model_dump(mode='python')
    # ... and process it
    output = {k: 2*v for k, v in params.items()}
    return output
