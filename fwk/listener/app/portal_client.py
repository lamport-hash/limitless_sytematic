# portal-client.py
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
from typing import Dict, Any

import uuid
import os
import requests
from pydantic import BaseModel, EmailStr
import bcrypt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()))

# Portal configuration

PORTAL_URL = os.getenv("PORTAL_URL", "http://localhost:8000")
SERVICE_NAME = os.getenv("SERVICE_NAME", os.getenv("HOSTNAME", "unknown-service"))
SERVICE_ID = os.getenv("SERVICE_ID", SERVICE_NAME)
SYSTEM_NAME = os.getenv("SYSTEM_NAME", "default-system")  # ← renamed
SERVICE_PATH = os.getenv("SERVICE_PATH", "/")
DESCRIPTION = os.getenv("SERVICE_DESCRIPTION", f"Service {SERVICE_NAME}")
REQUIRES_AUTH = os.getenv("REQUIRES_AUTH", "true").lower() == "true"
DOCKER_URL = os.getenv("DOCKER_URL", "DOCKER_URL env missing")
EXTERNAL_URL = os.getenv("EXTERNAL_URL", "EXTERNAL_URL env missing")
IS_DOCKER_SERVICE = os.getenv("IS_DOCKER_SERVICE", "IS_DOCKER_SERVICE env missing")

REGISTRATION_TOKEN = os.getenv(
    "REGISTRATION_TOKEN", "secure-registration-token-change-me"
)

# Database setup
#DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./portal_client.db")
DATABASE_URL =  "sqlite:///./portal_client.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def build_registration_config() -> Dict[str, Any]:
    """
    Build service registration payload.
    - 100% Docker / env-var driven
    - No config files
    - Works out-of-the-box with docker-compose, Docker Swarm, Kubernetes
    """
    # Smart defaults + env var overrides (all optional)
    service_name = SERVICE_NAME
    system_name = SYSTEM_NAME
    service_path = SERVICE_PATH
    description = DESCRIPTION
    requires_auth = REQUIRES_AUTH
    is_docker_service = IS_DOCKER_SERVICE

    # Used by other services inside the Docker network (DNS name)
    docker_url = DOCKER_URL
    external_url = EXTERNAL_URL

    return {
        "system_name": system_name,  # ← changed from "bundle_name"
        "services": [
            {
                "service_name": service_name,
                "service_path": service_path,
                "service_docker_url": f"{docker_url}",
                "service_ip_url": f"{external_url}",
                "description": description,
                "requires_auth": requires_auth,
                "is_docker_service": is_docker_service,
            }
        ],
    }


def register_service(portal_url: str, token: str):
    """
    Register the service with the portal by sending the registration config.
    """
    config = build_registration_config()
    response = requests.post(
        f"{portal_url}/portal/register-service",
        json=config,
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    return response.json()


def register():
    print(f"[{os.uname().nodename}] Starting service registration...")

    # Run the registration
    try:
        result = register_service(PORTAL_URL, REGISTRATION_TOKEN)
        print(f"Successfully registered with portal: {result}")
        return True
    except Exception as e:
        print(f"Failed to register with portal: {e}")
        print(
            "Continuing anyway (or change to sys.exit(1) if you want container to fail)"
        )
        # sys.exit(1)  # ← uncomment if registration must succeed
        return False


# Database models
class PortalClientUser(Base):
    __tablename__ = "portal_client_users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    portal_user_id = Column(
        String(255), nullable=False, index=True
    )  # Reference to portal user
    service_token = Column(
        String(500), nullable=False, index=True
    )  # Current service token
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class PortalClientSession(Base):
    __tablename__ = "portal_client_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False)  # Reference to portal_client_users.id
    session_token = Column(String(500), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    service_token = Column(String(500), nullable=False)  # Token used for this session


# Create tables
Base.metadata.create_all(bind=engine)


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_status_html(db: Session) -> str:
    user = db.query(PortalClientUser).first()

    if user:
        user_display_name = user.full_name or user.email
        return f"""
            <div class="flex items-center gap-4">
                <span class="text-green-600 font-medium text-sm">
                    Logged in as: {user_display_name}
                </span>
                <span id="register-result" class="text-sm"></span>
                <button id="register-btn" class="bg-blue-500 hover:bg-blue-700 text-white text-sm py-2 px-4 rounded cursor-pointer">
                    Register Service
                </button>
                <button id="logout-btn" class="bg-red-500 hover:bg-red-700 text-white text-sm py-2 px-4 rounded cursor-pointer">
                    Logout
                </button>
            </div>
        """
    else:
        return """
            <div class="flex items-center gap-4">
                <span class="text-gray-500 text-sm">Not logged in</span>
                <button id="register-btn" class="bg-blue-500 hover:bg-blue-700 text-white text-sm py-2 px-4 rounded cursor-pointer">
                    Register Service
                </button>
                <span id="register-result" class="text-sm"></span>
            </div>
        """


# Pydantic models for request/response
class UserLoginRequest(BaseModel):
    """Request model for login_with_portal_token endpoint"""

    user_data: dict  # Contains user info like email, full_name
    token: str  # Service token from portal


class LoginResponse(BaseModel):
    """Response model for successful login"""

    message: str
    user_id: str
    email: str
    dark_mode: bool
    service_token: str
    expires_in: int


async def verify_portal_service_token(authorization: Optional[str] = Header(None)):
    """Verify token with portal service"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")
    return await verify_token_with_portal(token)


async def verify_token_with_portal(token: str):
    """Verify a service token with the portal service"""
    try:
        # Decode token to get service_id
        import jwt

        payload = jwt.decode(
            token, "your-jwt-secret-change-in-production", algorithms=["HS256"]
        )
        service_id = payload.get("service_id")
        if not service_id:
            raise HTTPException(status_code=401, detail="Invalid token: no service_id")

        response = requests.post(
            f"{PORTAL_URL}/auth/verify-service-token",
            json={
                "token": token,
                "service_id": service_id,
            },
            timeout=5,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("valid"):
                return data

        raise HTTPException(
            status_code=401, detail=f"Invalid token - portal did reject {PORTAL_URL}"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Portal service unavailable: {str(e)}"
        )


async def verify_user_token_from_js(authorization: Optional[str] = Header(None)):
    """Verify user token passed from JavaScript frontend"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_token = authorization.replace("Bearer ", "")

    try:
        # Decode JWT token to get user_id
        import jwt

        payload = jwt.decode(
            user_token, "your-jwt-secret-change-in-production", algorithms=["HS256"]
        )
        user_id = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user_id")

        # Verify with portal
        response = requests.get(
            f"{PORTAL_URL}/auth/me",
            headers={"Authorization": f"Bearer {user_token}"},
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

        raise HTTPException(
            status_code=401, detail="Invalid user token - portal rejected"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Portal service unavailable: {str(e)}"
        )


router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}


@router.get("/register")
async def register_into_portal():
    try:
        success = register()

        if success:
            msg = SERVICE_NAME + " Service registered successfully with portal"
            return {
                "success": True,
                "message": msg,
            }
        else:
            msg = SERVICE_NAME + " Registration failed but service continues"
            return {
                "success": False,
                "message": msg,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Registration endpoint error",
        }


# Enhanced login_with_portal_token endpoint
@router.post("/login_with_portal_token")
async def login_with_portal_token(
    request: UserLoginRequest, db: Session = Depends(get_db)
):
    """
    Authenticate user using portal service token.

    Flow:
    1) Since the portal already verified the token when creating it,
       we trust the user data from the portal
    2) Authenticate the user and save the token in their profile
    3) Return success response with user info and service token
    """

    # Step 1: Extract user information from request (portal has already verified)
    user_email = request.user_data.get("email")
    user_full_name = request.user_data.get("full_name", "")
    portal_user_id = request.user_data.get("user_id", "")  # Portal should include this

    if not user_email:
        raise HTTPException(status_code=400, detail="User email is required")

    if not portal_user_id:
        raise HTTPException(status_code=400, detail="Portal user ID is required")

    # Step 3: Create or update user in portal_client service
    user = (
        db.query(PortalClientUser)
        .filter(
            (PortalClientUser.portal_user_id == portal_user_id)
            | (PortalClientUser.email == user_email)
        )
        .first()
    )

    if user:
        # Update existing user with latest portal data
        user.portal_user_id = portal_user_id  # Update if changed
        user.email = user_email
        user.full_name = user_full_name
        user.service_token = request.token
        user.updated_at = datetime.utcnow()
    else:
        # Create new user
        user = PortalClientUser(
            email=user_email,
            full_name=user_full_name,
            portal_user_id=portal_user_id,
            service_token=request.token,
        )
        db.add(user)

    db.commit()
    db.refresh(user)

    # Step 4: Create session
    session_token = f"portal_client_session_{uuid.uuid4()}"
    expires_at = datetime.utcnow() + timedelta(hours=1)

    session = PortalClientSession(
        user_id=user.id,
        session_token=session_token,
        expires_at=expires_at,
        service_token=request.token,
    )

    db.add(session)
    db.commit()

    # Step 5: Return success response
    return LoginResponse(
        message="Authenticated successfully",
        user_id=str(user.id),
        email=user.email,
        dark_mode=request.user_data.get("dark_mode", False),
        service_token=session_token,
        expires_in=3600,  # 1 hour
    )


# this is a demo of a protected end point
@router.get("/protected_by_service_token")
async def protected_by_service_token(
    user_data: dict = Depends(verify_portal_service_token),
):
    return {
        "message": "Token is valid",
        "user_info": {
            "user_id": user_data.get("user_id"),
            "email": user_data.get("email"),
        },
        "service": SERVICE_NAME,
    }


@router.get("/protected_by_internal_token")
async def protected_by_internal_token(db: Session = Depends(get_db)):
    """
    Protected endpoint that does not require a token from the client.
    Instead, it retrieves the service token stored in the portal_client database
    and verifies it with the portal.
    """
    user = db.query(PortalClientUser).first()

    if not user:
        raise HTTPException(status_code=401, detail="No user found in service database")

    if not user.service_token:
        raise HTTPException(
            status_code=401, detail="No service token associated with user"
        )

    try:
        user_data = await verify_token_with_portal(user.service_token)
        return {
            "message": "Token is valid",
            "user_info": {
                "user_id": user_data.get("user_id"),
                "email": user_data.get("email"),
            },
            "service": SERVICE_NAME,
        }
    except HTTPException as e:
        raise e


@router.get("/protected_by_user_token")
async def protected_by_user_token(
    user_data: dict = Depends(verify_user_token_from_js),
):
    """
    Protected endpoint that requires a user token passed from JavaScript frontend.
    The token is verified with the portal and user information is returned.
    """
    return {
        "message": "User token is valid",
        "user_info": {
            "user_id": user_data.get("id"),
            "email": user_data.get("email"),
            "full_name": user_data.get("full_name"),
        },
        "service": SERVICE_NAME,
    }


@router.get("/user-status-html")
async def user_status_html_endpoint(db: Session = Depends(get_db)):
    html = get_user_status_html(db)
    return {"html": html}
