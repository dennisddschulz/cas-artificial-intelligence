#!/usr/bin/env python3
"""
Weights & Biases Setup Helper Script
=====================================

This script helps you configure and authenticate with Weights & Biases.

Usage:
    python setup_wandb.py
    python setup_wandb.py --api-key YOUR_API_KEY
    python setup_wandb.py --check
"""

import sys
import os
import argparse
from pathlib import Path


def check_wandb_installed():
    """Check if wandb is installed."""
    try:
        import wandb
        return True, wandb.__version__
    except ImportError:
        return False, None


def install_wandb():
    """Install wandb package."""
    print("Installing wandb...")
    os.system("pip install wandb")
    return check_wandb_installed()


def check_authentication():
    """Check if user is authenticated with W&B."""
    try:
        import wandb
        api = wandb.Api()
        user = api.user()
        return True, user.username
    except Exception as e:
        return False, str(e)


def interactive_login():
    """Perform interactive login."""
    import wandb
    print("\n" + "="*60)
    print("INTERACTIVE WANDB LOGIN")
    print("="*60)
    print("\nYour browser will open to authenticate.")
    print("If it doesn't, visit: https://wandb.ai/authorize")
    print("\nOr paste your API key when prompted.")
    print("="*60 + "\n")

    try:
        wandb.login()
        authenticated, username = check_authentication()
        if authenticated:
            print(f"\n✓ Successfully logged in as: {username}")
            return True
        else:
            print("\n✗ Authentication failed. Please try again.")
            return False
    except Exception as e:
        print(f"\n✗ Login failed: {e}")
        return False


def set_api_key(api_key):
    """Set API key via environment variable."""
    print("\n" + "="*60)
    print("SETTING WANDB API KEY")
    print("="*60)

    os.environ['WANDB_API_KEY'] = api_key

    # Try to authenticate with the key
    import wandb
    try:
        wandb.login(key=api_key)
        authenticated, username = check_authentication()
        if authenticated:
            print(f"✓ Successfully authenticated with API key")
            print(f"✓ Logged in as: {username}")

            # Also save to .bashrc or .zshrc for persistence
            shell_config = Path.home() / ".bashrc"
            if not shell_config.exists():
                shell_config = Path.home() / ".zshrc"

            if shell_config.exists():
                with open(shell_config, 'a') as f:
                    f.write(f"\nexport WANDB_API_KEY='{api_key}'\n")
                print(f"✓ Added to {shell_config.name}")

            return True
        else:
            print("✗ Authentication failed with this API key")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def show_help():
    """Show help information."""
    print("""
WEIGHTS & BIASES SETUP HELPER
=============================

This script helps you configure Weights & Biases for the Trading RL project.

USAGE:
    python setup_wandb.py                 # Interactive setup
    python setup_wandb.py --api-key KEY   # Setup with API key
    python setup_wandb.py --check         # Check authentication status
    python setup_wandb.py --help          # Show this help

FIRST TIME SETUP:
    1. Get API key: https://wandb.ai/authorize
    2. Run: python setup_wandb.py --api-key YOUR_API_KEY
    3. Done! Your credentials are saved.

METHODS TO AUTHENTICATE:
    Method 1 (Interactive):
        python setup_wandb.py
        # Paste your API key when prompted
    
    Method 2 (Direct API Key):
        python setup_wandb.py --api-key eyJ0eXAiOiJKV1QiLCJhbGc...
    
    Method 3 (Environment Variable):
        export WANDB_API_KEY='your-api-key-here'
        python setup_wandb.py

TROUBLESHOOTING:
    - Can't find API key? Visit: https://wandb.ai/authorize
    - Still having issues? Check: WANDB_LOGIN_GUIDE.md

INFO:
    - Project: trading-rl-forecast
    - Dashboard: https://wandb.ai/YOUR-USERNAME/trading-rl-forecast
""")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup Weights & Biases for Trading RL Project",
        add_help=False
    )
    parser.add_argument('--api-key', type=str, help='API key for authentication')
    parser.add_argument('--check', action='store_true', help='Check authentication status')
    parser.add_argument('--help', action='store_true', help='Show help')

    args = parser.parse_args()

    if args.help:
        show_help()
        return 0

    # Check if wandb is installed
    installed, version = check_wandb_installed()
    if not installed:
        print("wandb is not installed. Installing now...")
        installed, version = install_wandb()
        if not installed:
            print("✗ Failed to install wandb. Please run: pip install wandb")
            return 1

    print(f"✓ wandb {version} is installed")

    # Check current authentication
    if args.check:
        print("\nChecking W&B authentication...")
        authenticated, info = check_authentication()
        if authenticated:
            print(f"✓ Authenticated as: {info}")
            return 0
        else:
            print(f"✗ Not authenticated: {info}")
            return 1

    # Set API key if provided
    if args.api_key:
        success = set_api_key(args.api_key)
        return 0 if success else 1

    # Interactive setup
    print("\n" + "="*60)
    print("WEIGHTS & BIASES SETUP")
    print("="*60)

    # Check if already authenticated
    authenticated, info = check_authentication()
    if authenticated:
        print(f"\n✓ Already authenticated as: {info}")
        return 0

    print("\n1. Get your API key: https://wandb.ai/authorize")
    print("\n2. Authenticate using one of these methods:\n")

    print("   a) Interactive (paste API key):")
    print("      Just press Enter below\n")

    print("   b) Paste API key:")
    api_key = input("Paste API key (or press Enter for interactive login): ").strip()

    if api_key:
        success = set_api_key(api_key)
        return 0 if success else 1
    else:
        success = interactive_login()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

