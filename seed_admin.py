import asyncio
import uuid
from db.base import AsyncSessionLocal
from api.models import User, Organisation, UserRole
from api.auth.service import hash_password

async def seed():
    async with AsyncSessionLocal() as session:
        # Create an organization
        org = Organisation(name="VisionX HQ")
        session.add(org)
        await session.flush()

        # Create the super admin user with the exact credentials from the screenshot
        user = User(
            email="shubhsonakiya86@gmail.com",
            password_hash=hash_password("Shubh@86"),
            role=UserRole.super_admin,
            org_id=org.id,
            is_active=True
        )
        session.add(user)
        await session.commit()
        print("Successfully seeded super admin: shubhsonakiya86@gmail.com")

if __name__ == "__main__":
    asyncio.run(seed())
