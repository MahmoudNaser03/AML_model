from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, str_min_length=1, str_max_length=256, allow_inf_nan=False, extra="forbid")
    timestamp: datetime = Field(description="ISO-8601 transaction timestamp")
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: float = Field(ge=0)
    receiving_currency: str
    amount_paid: float = Field(ge=0)
    payment_currency: str
    payment_format: str
