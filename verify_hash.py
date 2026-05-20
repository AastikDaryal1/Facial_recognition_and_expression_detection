import asyncio
from db.base import AsyncSessionLocal
from sqlalchemy import select
from api.models import User
from api.auth.service import verify_password

async def test_login():
    async with AsyncSessionLocal() as session:
        user = await session.scalar(select(User).where(User.email == "shubhsonakiya86@gmail.com"))
        if not user:
            print("User not found!")
            return
            
        is_valid = verify_password("Shubh@18", user.password_hash)
        print(f"Password verification for Shubh@18: {is_valid}")
        print(f"Hash in DB: {user.password_hash}")

if __name__ == "__main__":
    asyncio.run(test_login())
