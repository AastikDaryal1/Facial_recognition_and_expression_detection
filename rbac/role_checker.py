from fastapi import Depends, HTTPException
from auth.dependencies import get_current_user

def require_role(allowed_roles: list):
    def checker(user=Depends(get_current_user)):
        if user.get("role") not in allowed_roles:
            raise HTTPException(status_code=403, detail="Access denied")
        return user
    return checker