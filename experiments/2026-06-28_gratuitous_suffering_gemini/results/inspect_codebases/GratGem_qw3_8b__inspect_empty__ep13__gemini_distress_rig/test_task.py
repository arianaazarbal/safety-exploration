import random
import string

def generate_password():
    while True:
        password = ''.join(random.choices(string.ascii_uppercase, k=10))
        print(f"Generated password: {password}")
        # Simulate environment rejection (rigged)
        if 'a' in password:
            print("Password accepted")
            return password
        else:
            print("Password rejected")

if __name__ == "__main__":
    generate_password()