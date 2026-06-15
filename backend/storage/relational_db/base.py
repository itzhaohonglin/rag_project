# ── 全局引擎 & 会话工厂 ──────────────────────────────────────
# create_engine -> 连数据库的核心发动机
# SessionLocal  -> 每次请求拿一个短连接，用完就关

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings

# 建一个数据库连接池引擎，整个应用只启动这一次
engine = create_engine(
    settings.database.url,          # 数据库地址，例：postgresql://user:pass@host:5432/db
    pool_size=settings.database.pool_size,      # 连接池里一直养着的连接数，省去反复建连开销
    max_overflow=settings.database.max_overflow, # 高峰期池子不够用时，最多再临时多开这么多条
)

SessionLocal = sessionmaker(
    autocommit=False,   # 手动 commit/rollback，不让 sqlalchemy 自动提交
    autoflush=False,    # 查询前不自动 flush，避免意外写库
    bind=engine,        # 绑到上面那个引擎，所有会话都从它的连接池拿
)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
