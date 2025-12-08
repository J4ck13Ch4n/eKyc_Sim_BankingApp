from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import models
from database import engine, get_db
import zkp
import uuid
import time

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="eKyc ZKP Server")

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
    # 1. Check if ID already enrolled
    if db.query(models.User).filter(models.User.id_hash == payload.idNumberHash).first():
        raise HTTPException(status_code=400, detail="ID already enrolled")
    
    # 2. Reconstruct message
    # Message format must match Client: "ENROLL:commitment:id:name:dob:approval:ts"
    # Wait, looking at ZKPEnrollmentManager.kt:
    # "ENROLL:" + "commitment:$commitment:" + "id:$idNumberHash:" + ...
    # Let's double check the exact format in ZKPEnrollmentManager.kt
    
    # From ZKPEnrollmentManager.kt:
    # return "ENROLL:" +
    #         "commitment:$commitment:" +
    #         "id:$idNumberHash:" +
    #         "name:$fullNameHash:" +
    #         "dob:$dobHash:" +
    #         "approval:$approval:" +
    #         "ts:$timestamp"
            
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
    print(f"Commitment: {payload.commitment}")
    print(f"ID Hash: {payload.idNumberHash}")
    print(f"Encrypted PII: {payload.encryptedPII}")
    print(f"Proof: {proof_dict}")
    print(f"------------------------------------------")

    # 4. Save to DB
    new_user = models.User(
        public_key=payload.publicKey,
        commitment=payload.commitment,
        id_hash=payload.idNumberHash,
        encrypted_pii=payload.encryptedPII,
        enrollment_proof=proof_dict
    )
    
    print(f"--- [SERVER] Storing User to DB ---")
    print(f"User ID Hash: {new_user.id_hash}")
    print(f"User Public Key: {new_user.public_key}")
    print(f"-----------------------------------")

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"success": True, "userId": new_user.id}

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
    This simulates a privacy-preserving query where the government checks 
    if 'Hash(ID)' exists in the bank's database without revealing the ID to the bank 
    (if the bank didn't already have it) or dumping the whole DB.
    """
    user = db.query(models.User).filter(models.User.id_hash == id_hash).first()
    
    if user:
        return {"exists": True, "user_id": user.id, "created_at": user.created_at}
    else:
        return {"exists": False}


def detect(img):
    return random.choice([True, False])