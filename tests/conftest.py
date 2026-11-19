import pytest

from twtrend.config import Config
from twtrend.data import synthetic
from twtrend.data.loader import load_market_data
from twtrend.db import Database


@pytest.fixture(scope="session")
def db():
    d = Database(":memory:")
    synthetic.write_to_db(d, synthetic.generate(n_stocks=30, start="2016-01-01", end="2019-12-31", seed=3))
    return d


@pytest.fixture(scope="session")
def md(db):
    return load_market_data(db)


@pytest.fixture
def cfg():
    return Config()
