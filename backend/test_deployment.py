import requests
import time
import uuid
import hashlib
from ecdsa import SigningKey, SECP256k1
from zkp import compute_challenge, point_to_hex, int_to_hex

# Configuration
BASE_URL = "https://ekyc-backend-436637848640.asia-northeast1.run.app"
# BASE_URL = "http://localhost:8000" # Uncomment to test locally

def generate_key_pair():
    sk = SigningKey.generate(curve=SECP256k1)
    pk = sk.verifying_key
    return sk, pk

def create_enrollment_payload():
    print("Generating keys and proof...")
    sk, pk = generate_key_pair()
    
    # 1. Prepare Data
    id_number = f"ID_{uuid.uuid4()}"
    full_name = "Test User"
    dob = "1990-01-01"
    timestamp = int(time.time() * 1000)
    approval = 1 # Approved
    
    # Hashes
    id_hash = hashlib.sha256(id_number.encode()).hexdigest()
    name_hash = hashlib.sha256(full_name.encode()).hexdigest()
    dob_hash = hashlib.sha256(dob.encode()).hexdigest()
    
    # 2. Create ZKP Proof (Schnorr)
    # Commitment: R = r * G
    r = SigningKey.generate(curve=SECP256k1)
    R = r.verifying_key.pubkey.point
    
    # Message to sign
    # "ENROLL:commitment:id:name:dob:approval:ts"
    commitment_hex = point_to_hex(R)
    
    message = (f"ENROLL:commitment:{commitment_hex}:"
               f"id:{id_hash}:"
               f"name:{name_hash}:"
               f"dob:{dob_hash}:"
               f"approval:{approval}:"
               f"ts:{timestamp}")
               
    # Challenge c = Hash(R, P, message)
    c = compute_challenge(R, pk.pubkey.point, message)
    
    # Response s = r + c * x
    x = int.from_bytes(sk.to_string(), byteorder='big')
    s = (int.from_bytes(r.to_string(), byteorder='big') + c * x) % SECP256k1.order
    
    proof = {
        "commitmentR": commitment_hex,
        "challenge": int_to_hex(c),
        "response": int_to_hex(s)
    }
    
    # 3. Encrypted PII (Dummy)
    encrypted_pii = "dummy_encrypted_data"
    
    payload = {
        "publicKey": point_to_hex(pk.pubkey.point),
        "commitment": commitment_hex, # Note: In my main.py logic, it seems I used R as commitment?
        # Re-reading main.py: 
        # message = f"ENROLL:commitment:{payload.commitment}..."
        # And R comes from proof.commitmentR. 
        # In this scheme, usually commitment IS commitmentR.
        "idNumberHash": id_hash,
        "encryptedPII": encrypted_pii,
        "proof": proof,
        "timestamp": timestamp,
        "fullNameHash": name_hash,
        "dobHash": dob_hash,
        "approval": approval,
        "email": f"test_{uuid.uuid4()}@example.com",
        "password": "password123"
    }
    
    return payload, id_hash

def log(msg):
    print(msg)
    with open("test_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def test_enrollment():
    log(f"Testing Enrollment against {BASE_URL}...")
    
    try:
        payload, id_hash = create_enrollment_payload()
        
        response = requests.post(f"{BASE_URL}/api/enroll", json=payload)
        
        log(f"Status Code: {response.status_code}")
        log(f"Response: {response.text}")
        
        if response.status_code == 200:
            log("✅ Enrollment SUCCESS!")
            return id_hash
        else:
            log("❌ Enrollment FAILED")
            return None
            
    except Exception as e:
        log(f"❌ Error: {e}")
        return None

def test_check_citizen(id_hash):
    if not id_hash:
        print("Skipping check citizen test due to previous failure.")
        return

    print(f"\nTesting Check Citizen (DB Read) for {id_hash}...")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/check_citizen/{id_hash}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200 and response.json().get("exists") == True:
            print("✅ Check Citizen SUCCESS! Data persisted in DB.")
        else:
            print("❌ Check Citizen FAILED or User not found.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check if requests is installed
    try:
        import requests
    except ImportError:
        print("Please install requests: pip install requests")
        exit(1)

    id_hash = test_enrollment()
    test_check_citizen(id_hash)
