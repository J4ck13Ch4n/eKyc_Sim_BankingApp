from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import models
from database import engine, get_db
import zkp
import uuid
import time
from passlib.context import CryptContext

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="eKyc ZKP Server")

# Password Hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)


# Pydantic Models
class ProofData(BaseModel):
    commitmentR: str
    challenge: str
    response: str

class EnrollmentPayload(BaseModel):
    publicKey: str
    commitment: str
    idNumberHash: str
    encryptedPII: str
    proof: ProofData
    timestamp: int
    fullNameHash: str
    dobHash: str
    approval: int
    email: str     # NEW
    password: str  # NEW

class LoginPayload(BaseModel):
    email: str
    password: str

class VerificationPayload(BaseModel):
    publicKey: str
    proof: ProofData
    nullifier: str
    timestamp: int

# In-memory session store (for simplicity)
# In prod, use Redis
sessions = {}

@app.get("/api/challenge")
def get_challenge():
    """Generate a random session ID for login"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = int(time.time() * 1000)
    return {"sessionId": session_id}

@app.post("/api/enroll")
def enroll(payload: EnrollmentPayload, db: Session = Depends(get_db)):
    try:
        # 1. Check if ID already enrolled
        if db.query(models.User).filter(models.User.id_hash == payload.idNumberHash).first():
            raise HTTPException(status_code=400, detail="ID already enrolled")

        # 1.1 Check if Email already enrolled
        # This might fail if email column doesn't exist
        if db.query(models.User).filter(models.User.email == payload.email).first():
                raise HTTPException(status_code=400, detail="Email already used")
        
        # 2. Reconstruct message
        message = (f"ENROLL:commitment:{payload.commitment}:"
                   f"id:{payload.idNumberHash}:"
                   f"name:{payload.fullNameHash}:"
                   f"dob:{payload.dobHash}:"
                   f"approval:{payload.approval}:"
                   f"ts:{payload.timestamp}")
                   
        # 3. Verify Proof
        proof_dict = {
            "commitmentR": payload.proof.commitmentR,
            "challenge": payload.proof.challenge,
            "response": payload.proof.response
        }
        
        if not zkp.verify_proof(payload.publicKey, proof_dict, message):
            raise HTTPException(status_code=400, detail="Invalid ZKP Proof")
            
        # Log received payload
        print(f"--- [SERVER] Received Enrollment Payload ---")
        print(f"Public Key: {payload.publicKey}")
        print(f"Email: {payload.email}")
        print(f"Proof: {proof_dict}")
        print(f"Password length: {len(payload.password)}")
        print(f"------------------------------------------")

        # 4. Save to DB
        hashed_password = get_password_hash(payload.password)
        
        new_user = models.User(
            public_key=payload.publicKey,
            commitment=payload.commitment,
            id_hash=payload.idNumberHash,
            encrypted_pii=payload.encryptedPII,
            enrollment_proof=proof_dict,
            email=payload.email,
            password_hash=hashed_password
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {"success": True, "userId": new_user.id}
    except Exception as e:
        print(f"Enrollment Error: {e}")
        db.rollback()
        # RETURN THE ERROR FOR DEBUGGING
        raise HTTPException(status_code=500, detail=f"Enrollment Logic Error: {str(e)}. Pwd Len: {len(payload.password)}")

from sqlalchemy import inspect
@app.get("/api/admin/check_schema")
def check_schema():
    insp = inspect(engine)
    columns = insp.get_columns("users")
    return {"columns": [c["name"] for c in columns]}


@app.post("/api/login")
def login(payload: LoginPayload, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Email not found")
        
    if not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect password")
        
    return {"success": True, "userId": user.id, "detail": "Login successful"}

@app.post("/api/verify")
def verify(payload: VerificationPayload, sessionId: str, db: Session = Depends(get_db)):
    # 1. Check session validity (simple check)
    if sessionId not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired session")
        
    # 2. Check nullifier (Replay Attack Prevention)
    if db.query(models.VerificationLog).filter(models.VerificationLog.nullifier == payload.nullifier).first():
        raise HTTPException(status_code=400, detail="Replay attack detected (Nullifier used)")
        
    # 3. Get User
    user = db.query(models.User).filter(models.User.public_key == payload.publicKey).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # 4. Verify Proof
    # Message: "VERIFY:$sessionId:$timestamp"
    message = f"VERIFY:{sessionId}:{payload.timestamp}"
    
    proof_dict = {
        "commitmentR": payload.proof.commitmentR,
        "challenge": payload.proof.challenge,
        "response": payload.proof.response
    }
    
    if not zkp.verify_proof(payload.publicKey, proof_dict, message):
        raise HTTPException(status_code=400, detail="Invalid ZKP Proof")
        
    # 5. Log verification (consume nullifier)
    log = models.VerificationLog(
        nullifier=payload.nullifier,
        proof=proof_dict
    )
    db.add(log)
    db.commit()
    
    # Remove session
    del sessions[sessionId]
    
    return {"success": True, "userId": user.id}

@app.get("/api/admin/check_citizen/{id_hash}")
def check_citizen(id_hash: str, db: Session = Depends(get_db)):
    """
    [GOVERNMENT/ADMIN DEMO] Check if a citizen exists based on ID Hash.
    """
    user = db.query(models.User).filter(models.User.id_hash == id_hash).first()
    
    if user:
        return {"exists": True, "user_id": user.id, "created_at": user.created_at}
    else:
        return {"exists": False}

from sqlalchemy import text
@app.post("/api/admin/migrate")
def migrate(db: Session = Depends(get_db)):
    """
    Temporary endpoint to migrate DB schema (add email and password_hash)
    """
    try:
        # Check if column exists first to avoid error? Or just try-catch.
        # Postgres supports IF NOT EXISTS for adding columns in newer versions, 
        # but let's just try.
        
        # 1. Add email
        try:
            db.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR"))
            db.execute(text("CREATE UNIQUE INDEX ix_users_email ON users (email)"))
            print("Added email column")
        except Exception as e:
            print(f"Email column might already exist: {e}")
            db.rollback()
            
        # 2. Add password_hash
        try:
            db.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR"))
            print("Added password_hash column")
        except Exception as e:
            print(f"Password_hash column might already exist: {e}")
            db.rollback()
            
        db.commit()
        return {"message": "Migration attempted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

