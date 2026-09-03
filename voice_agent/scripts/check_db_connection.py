"""
Изолированная проверка подключения к Postgres+pgvector через shared/db.py,
до того как строить вокруг него SQLAlchemy-модели и MCP-тулы.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from shared.db import engine


async def main() -> None:
    async with engine.connect() as conn:
        version = await conn.scalar(text("SELECT version();"))
        print("Подключение установлено.")
        print("Postgres:", version)

        vector_version = await conn.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        )
        print("pgvector extension version:", vector_version)

        result = await conn.execute(text("SELECT '[1,2,3]'::vector;"))
        print("Тестовый vector-литерал:", result.scalar())


if __name__ == "__main__":
    asyncio.run(main())
