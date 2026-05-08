import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import base64
import hashlib
from dotenv import load_dotenv

# Load .env file from multiple possible locations
load_dotenv()  # Current directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))  # Parent directory

SECRET_KEY = os.getenv('SESSION_SECRET')
if not SECRET_KEY:
    raise RuntimeError("SESSION_SECRET environment variable is required for security")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_encryption_key():
    key = hashlib.sha256(SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(key)

def encrypt_credential(credential: str) -> str:
    if not credential:
        return None
    f = Fernet(get_encryption_key())
    return f.encrypt(credential.encode()).decode()

def decrypt_credential(encrypted: str) -> str:
    if not encrypted:
        print(f"[DECRYPT] No encrypted value provided", flush=True)
        return None
    try:
        f = Fernet(get_encryption_key())
        result = f.decrypt(encrypted.encode()).decode()
        print(f"[DECRYPT] Success, length: {len(result)}", flush=True)
        return result
    except Exception as e:
        print(f"[DECRYPT] ERROR: {e}", flush=True)
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
