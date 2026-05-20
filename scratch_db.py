import asyncio
from db.base import AsyncSessionLocal
from sqlalchemy import select
from api.models import User

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User.email, User.role, User.is_active))
        users = result.all()
        print("--- USER LIST ---")
        for email, role, is_active in users:
            print(f"Email: {email} | Role: {role.value} | Active: {is_active}")
        print("-----------------")

if __name__ == "__main__":
    asyncio.run(main())
