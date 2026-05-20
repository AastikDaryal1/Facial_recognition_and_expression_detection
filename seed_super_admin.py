import asyncio
from db.base import AsyncSessionLocal
from sqlalchemy import text
from api.models import User, Organisation, UserRole
from api.auth.service import hash_password

async def seed_super_admin():
    async with AsyncSessionLocal() as session:
        # Purge any old stale admin data to ensure clean state
        await session.execute(text("DELETE FROM users"))
        await session.execute(text("DELETE FROM organisations"))
        await session.commit()
        
        # Create primary Organisation
        org = Organisation(name="Netsmartz")
        session.add(org)
        await session.flush()
        
        # Create Super Admin User
        user = User(
            email="shubhsonakiya86@gmail.com",
            password_hash=hash_password("Shubh@86"),
            role=UserRole.super_admin,
            org_id=org.id
        )
        session.add(user)
        await session.commit()
        print("Successfully seeded Super Admin 'shubhsonakiya86@gmail.com' with password 'Shubh@86'.")

if __name__ == "__main__":
    asyncio.run(seed_super_admin())
