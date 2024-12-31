from fastapi import HTTPException, status


class ServerError(HTTPException):
    def __init__(self, detail: str = "Server error."):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
