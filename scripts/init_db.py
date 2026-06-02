"""Initialize database tables."""

from backend.storage.relational_db.base import engine
from backend.storage.relational_db.models import Base


def init():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")


if __name__ == "__main__":
    init()
