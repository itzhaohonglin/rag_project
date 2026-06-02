import pytest


@pytest.fixture
def milvus_config():
    return {"host": "localhost", "port": 19530, "collection": "test_collection"}


@pytest.fixture
def db_url():
    return "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_project_test"
