import asyncio
from db.base import AsyncSessionLocal
from sqlalchemy import text
from api.models import User, Organisation

async def purge_db():
    async with AsyncSessionLocal() as session:
        # Delete all users and organizations to reset the setup state
        await session.execute(text("DELETE FROM users"))
        await session.execute(text("DELETE FROM organisations"))
        await session.commit()
        print("Successfully purged all users and organisations. Setup state is now reset.")

if __name__ == "__main__":
    asyncio.run(purge_db())
