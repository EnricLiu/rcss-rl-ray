from pydantic import BaseModel, ConfigDict, Field

from rcss_env.bhv import BhvMixin, BhvHeliosView, BhvNaiveNeck


class BhvConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    neck: BhvMixin = Field(default_factory=lambda: BhvNaiveNeck())
    view: BhvMixin = Field(default_factory=lambda: BhvHeliosView())
