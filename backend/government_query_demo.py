import requests
import hashlib
import json
import os
import sys

# Configuration
API_URL = "https://ekyc-backend-436637848640.asia-northeast1.run.app"  # Cloud Run Deployment

def sha256(data: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def query_citizen_status(id_number: str, full_name: str):
    """
    Simulate a government query to check if a citizen has a bank account.
    """
    print(f"\nScanning Database for: {full_name} (ID: {id_number})...")
    
    # 1. Compute the hash of the ID number
    id_hash = sha256(id_number)
    
    try:
        response = requests.get(f"{API_URL}/api/admin/check_citizen/{id_hash}")
        
        if response.status_code == 200:
            data = response.json()
            if data["exists"]:
                print(f"\n[RESULT] ✅ FOUND MATCH!")
                print(f"--------------------------------------------------")
                print(f"Citizen Name  : {full_name}")
                print(f"Citizen ID    : {id_number}")
                print(f"Bank Account  : LINKED")
                print(f"Ref User ID   : {data['user_id']}")
                print(f"Registered At : {data['created_at']}")
                print(f"--------------------------------------------------")
            else:
                print(f"\n[RESULT] ❌ NO RECORD FOUND")
                print(f"--------------------------------------------------")
                print(f"Citizen {full_name} (ID: {id_number}) does NOT have a bank account.")
                print(f"--------------------------------------------------")
        else:
            print(f"Error querying bank API: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the backend server availability.")

def print_header():
    print("==================================================")
    print("          GOVERNMENT CITIZEN QUERY TOOL           ")
    print("==================================================")
    print("This tool simulates a government agency finding")
    print("if a citizen has an account at the eKYC Bank.")
    print("==================================================\n")

def main():
    while True:
        clear_screen()
        print_header()
        
        print("1. Query Citizen Status")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == '1':
            print("\n--- ENTER CITIZEN DETAILS ---")
            full_name = input("Full Name: ").strip()
            id_number = input("ID Number: ").strip()
            
            if not full_name or not id_number:
                print("Error: Name and ID cannot be empty.")
            else:
                query_citizen_status(id_number, full_name)
            
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            print("\nExiting tool. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting...")
        sys.exit(0)
