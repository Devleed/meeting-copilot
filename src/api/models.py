from pydantic import BaseModel


class SuggestRequest(BaseModel):
    speech: str
    manual: bool = False


class SuggestResponse(BaseModel):
    response: str
