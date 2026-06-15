"""Initialize database tables."""

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，使 from backend.xxx 能找到包
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.relational_db.base import engine
from backend.storage.relational_db.models import Base


def init():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")


if __name__ == "__main__":
    init()
