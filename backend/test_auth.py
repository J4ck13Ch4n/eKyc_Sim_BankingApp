import requests
import time
import uuid
import secrets
import zkp  # Local import

BASE_URL = "http://localhost:8000"

def generate_random_hex(length=64):
    return secrets.token_hex(length // 2)

def generate_proof_local(priv_key_int, message):
    # Mimic SchnorrZKP.generateProof
    # 1. Generate random k (ephemeral key)
    k = secrets.randbelow(zkp.order)
    
    # 2. Compute commitment R = k*G
    R = zkp.generator * k
    
    # 3. Compute public key P = x*G
    P = zkp.generator * priv_key_int
    
    # 4. Compute challenge c = Hash(R || P || message)
    # Use the server's function to ensure match
    c = zkp.compute_challenge(R, P, message)
    
    # 5. Compute response s = k + c*x (mod n)
    s = (k + (c * priv_key_int)) % zkp.order
    
    return {
        "commitmentR": zkp.point_to_hex(R),
        "challenge": zkp.int_to_hex(c),
        "response": zkp.int_to_hex(s)
    }

def test_auth_flow():
    print("--- Starting Auth Flow Test ---")
    
    # 1. Setup Identity
    try:
        priv_key = secrets.randbelow(zkp.order)
        pub_key_point = zkp.generator * priv_key
        pub_key_hex = zkp.point_to_hex(pub_key_point)
    except Exception as e:
        print(f"Failed to generate keys: {e}")
        return

    # 2. Enroll
    email = f"test_{uuid.uuid4()}@example.com"
    password = "MySecurePassword123"
    
    # Message construction MUST match server exact format in main.py
    commitment = generate_random_hex(64) # This is usually Hash(pubKey || idHash) but random hex works for format
    id_hash = generate_random_hex(64)
    name_hash = generate_random_hex(64)
    dob_hash = generate_random_hex(64)
    approval = 1
    timestamp = int(time.time() * 1000)
    
    message = (f"ENROLL:commitment:{commitment}:"
               f"id:{id_hash}:"
               f"name:{name_hash}:"
               f"dob:{dob_hash}:"
               f"approval:{approval}:"
               f"ts:{timestamp}")
               
    # Generate Proof
    proof_data = generate_proof_local(priv_key, message)
    
    payload = {
        "publicKey": pub_key_hex,
        "commitment": commitment,
        "idNumberHash": id_hash,
        "encryptedPII": "{}",
        "proof": proof_data,
        "timestamp": timestamp,
        "fullNameHash": name_hash,
        "dobHash": dob_hash,
        "approval": approval,
        "email": email,
        "password": password
    }
    
    print(f"Registering with {email}...")
    try:
        resp = requests.post(f"{BASE_URL}/api/enroll", json=payload)
        print(f"Enroll Response: {resp.status_code} - {resp.text}")
        
        if resp.status_code == 200:
            print("Enrollment Successful.")
            
            # 3. Login
            print("Logging in...")
            login_payload = {
                "email": email,
                "password": password
            }
            resp_login = requests.post(f"{BASE_URL}/api/login", json=login_payload)
            print(f"Login Response: {resp_login.status_code} - {resp_login.text}")
            
            if resp_login.status_code != 200:
                print("Login FAILED!")
                return # Fail early

            # 4. Login with wrong password
            print("Logging in with wrong password...")
            login_payload["password"] = "wrongpass"
            resp_wrong = requests.post(f"{BASE_URL}/api/login", json=login_payload)
            print(f"Wrong Password Response: {resp_wrong.status_code} - {resp_wrong.text}")
            
            if resp_wrong.status_code == 400:
                print("Wrong password check PASSED.")
            else:
                print("Wrong password check FAILED.")
                
        else:
            print("Skipping Login test due to Enroll failure.")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Connection refused. Is the uvicorn server running?")

if __name__ == "__main__":
    test_auth_flow()
