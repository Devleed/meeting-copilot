from pydantic import BaseModel


class SuggestRequest(BaseModel):
    text: str
    manual: bool = False


class SuggestResponse(BaseModel):
    response: str
