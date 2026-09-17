"""REST API over the Business Language Layer.

A thin FastAPI wrapper around ``olist.bll.PredictionService``. No modelling,
explanation or business logic is reimplemented here.

    uvicorn olist.api.app:app --reload

The FastAPI instance is deliberately *not* re-exported as ``olist.api.app``:
that name belongs to the submodule, and binding the instance to it shadows
the module for anyone doing ``import olist.api.app``. Import it explicitly::

    from olist.api.app import app
"""

__all__: list[str] = []
