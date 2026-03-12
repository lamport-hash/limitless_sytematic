#!/usr/bin/env python3
import secrets
import string


def generate_api_key(length=24):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


if __name__ == "__main__":
    print("Generated API Keys:")
    print(f"  Admin key:  mgr-{generate_api_key()}")
    print(f"  Worker key: worker-{generate_api_key()}")
    print(f"  Client key: client-{generate_api_key()}")
