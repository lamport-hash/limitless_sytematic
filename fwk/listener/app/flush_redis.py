#!/usr/bin/env python3
"""
Script to completely flush all Redis data for the singularity_listener.
This will delete ALL keys in the Redis database.
"""

import redis
import sys
import os


def main():
    # Redis connection parameters - adjust as needed
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB", 0))

    try:
        # Connect to Redis
        r = redis.Redis(host=host, port=port, db=db, decode_responses=False)

        # Test connection
        r.ping()
        print(f"Connected to Redis at {host}:{port}, db {db}")

        # Get info before flush
        info_before = r.info("keyspace")
        total_keys_before = (
            sum(db_info.get("keys", 0) for db_info in info_before.values())
            if info_before
            else 0
        )
        print(f"Keys before flush: {total_keys_before}")

        # Confirm with user (skip if --force flag is used)
        import argparse

        parser = argparse.ArgumentParser(description="Flush Redis database")
        parser.add_argument(
            "--force", action="store_true", help="Skip confirmation prompt"
        )
        args = parser.parse_args()

        if not args.force:
            try:
                confirm = input(
                    "This will delete ALL data in Redis. Are you sure? (type 'yes' to confirm): "
                )
                if confirm.lower() != "yes":
                    print("Operation cancelled.")
                    sys.exit(0)
            except EOFError:
                print(
                    "Non-interactive mode detected. Use --force to skip confirmation."
                )
                sys.exit(1)

        # Flush all data (all databases)
        result = r.flushall()
        print(f"Flush result: {result}")

        # Get info after flush
        info_after = r.info("keyspace")
        total_keys_after = (
            sum(db_info.get("keys", 0) for db_info in info_after.values())
            if info_after
            else 0
        )
        print(f"Keys after flush: {total_keys_after}")

        if total_keys_after == 0:
            print("✅ Redis database successfully flushed - all data deleted.")
        else:
            print("⚠️  Warning: Some keys may still exist.")

    except redis.ConnectionError as e:
        print(f"❌ Failed to connect to Redis: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
