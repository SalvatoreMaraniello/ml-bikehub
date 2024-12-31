from pydantic import BaseModel, Field


class InputModel(BaseModel):
    avg_duration_prev_7days: float = Field(
        None, description='Avg. duration of trips starting from same dockstation.', ge=0.0,
        example=13.5)
    HPCP: float = Field(
        None, description='Total precipitation over previous hour', ge=0.0, example=0.2
    )
    dow: int = Field(
        None, ge=0, le=6, description='Day of the week, starting from 0 (Monday) to 6 (Sunday).',
        example=5)
    is_registered: bool = Field(
        None, description='Whether the user is registered or not.', example=True,
    )
    has_trace: bool = Field(
        None, description='Whether there are traces of rain.', example=False,
    )


class OutputModel(BaseModel):
    '''A test model to verify communication with API works as intended'''
    duration: float = Field(description='Trip duration, in minutes.')


class TestModel(BaseModel):
    '''A test model to verify communication with API works as intended'''
    param_int: int = Field(description='An integer.')
    param_str: str = Field(description='A string.')


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    payload = InputModel(
        avg_duration_prev_7days=13.0,
        HPCP=0.0,
        dow=4,
        is_registered=True,
        has_trace=False
    )

    # transform to disctionary
    payload_dict = payload.dict()
    dow = payload_dict.pop('dow')

    df_payload = pd.DataFrame([payload_dict])
    df_payload['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df_payload['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    df_payload['is_registered'] = df_payload['is_registered'].astype(int)
    df_payload['has_trace'] = df_payload['has_trace'].astype(int)
