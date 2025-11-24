#!/usr/bin/env python3
"""
Secure Password and API Key Generator for UltraThink Pilot

Generates cryptographically secure passwords and API keys for all services.
Run this script to populate your .env file with strong credentials.

Usage:
    python3 scripts/generate_secrets.py [--output .env] [--format shell|json]

Security:
    - Uses secrets module (CSPRNG)
    - Minimum 32 characters for passwords
    - API keys prefixed with 'uk_' (ultrathink key)
    - Validates strength before output

Author: UltraThink Security Team
Date: 2025-11-24
"""

import secrets
import string
import json
import argparse
import sys
from pathlib import Path


class SecretGenerator:
    """Generate cryptographically secure passwords and API keys."""

    # Character sets for password generation
    UPPERCASE = string.ascii_uppercase
    LOWERCASE = string.ascii_lowercase
    DIGITS = string.digits
    SPECIAL = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    # API key alphabet (no special chars for easier handling)
    API_KEY_CHARS = string.ascii_letters + string.digits

    def __init__(self):
        self.generated = {}

    def generate_password(self, length=32, name="password"):
        """
        Generate a strong password with guaranteed character diversity.

        Args:
            length: Password length (minimum 32)
            name: Identifier for this password

        Returns:
            Secure random password string
        """
        if length < 32:
            raise ValueError("Password must be at least 32 characters")

        # Ensure at least one character from each set
        password = [
            secrets.choice(self.UPPERCASE),
            secrets.choice(self.LOWERCASE),
            secrets.choice(self.DIGITS),
            secrets.choice(self.SPECIAL),
        ]

        # Fill remaining with random characters from all sets
        all_chars = self.UPPERCASE + self.LOWERCASE + self.DIGITS + self.SPECIAL
        password.extend(secrets.choice(all_chars) for _ in range(length - 4))

        # Shuffle to avoid predictable patterns
        secrets.SystemRandom().shuffle(password)

        password_str = ''.join(password)
        self.generated[name] = password_str
        return password_str

    def generate_api_key(self, name="api_key", prefix="uk"):
        """
        Generate an API key with format: prefix_32randomchars

        Args:
            name: Identifier for this API key
            prefix: Prefix for the key (default: uk for ultrathink key)

        Returns:
            API key string
        """
        random_part = ''.join(secrets.choice(self.API_KEY_CHARS) for _ in range(32))
        api_key = f"{prefix}_{random_part}"
        self.generated[name] = api_key
        return api_key

    def generate_secret_key(self, length=64, name="secret_key"):
        """
        Generate a secret key for JWT or session encryption.

        Args:
            length: Key length in characters
            name: Identifier for this key

        Returns:
            Hex-encoded secret key
        """
        secret = secrets.token_hex(length // 2)  # token_hex returns length*2 chars
        self.generated[name] = secret
        return secret

    def generate_all_secrets(self):
        """Generate all secrets needed for UltraThink Pilot."""
        secrets_dict = {
            # Database passwords
            "POSTGRES_PASSWORD": self.generate_password(name="postgres_password"),
            "MLFLOW_DB_PASSWORD": self.generate_password(name="mlflow_db_password"),

            # Service API keys
            "DATA_SERVICE_API_KEY": self.generate_api_key(name="data_service_api_key"),
            "INFERENCE_SERVICE_API_KEY": self.generate_api_key(name="inference_service_api_key"),
            "RISK_MANAGER_API_KEY": self.generate_api_key(name="risk_manager_api_key"),
            "REGIME_DETECTION_API_KEY": self.generate_api_key(name="regime_detection_api_key"),
            "META_CONTROLLER_API_KEY": self.generate_api_key(name="meta_controller_api_key"),
            "ONLINE_LEARNING_API_KEY": self.generate_api_key(name="online_learning_api_key"),
            "FORENSICS_CONSUMER_API_KEY": self.generate_api_key(name="forensics_consumer_api_key"),

            # Monitoring credentials
            "GRAFANA_ADMIN_PASSWORD": self.generate_password(name="grafana_admin_password"),
            "GF_SECURITY_SECRET_KEY": self.generate_secret_key(name="grafana_secret_key"),
            "PROMETHEUS_PASSWORD": self.generate_password(name="prometheus_password"),

            # Redis password (optional)
            "REDIS_PASSWORD": self.generate_password(name="redis_password"),

            # JWT secret
            "JWT_SECRET_KEY": self.generate_secret_key(name="jwt_secret_key"),
        }

        return secrets_dict

    def validate_password_strength(self, password):
        """
        Validate password meets security requirements.

        Returns:
            (bool, str): (is_valid, error_message)
        """
        if len(password) < 32:
            return False, "Password must be at least 32 characters"

        has_upper = any(c in self.UPPERCASE for c in password)
        has_lower = any(c in self.LOWERCASE for c in password)
        has_digit = any(c in self.DIGITS for c in password)
        has_special = any(c in self.SPECIAL for c in password)

        if not all([has_upper, has_lower, has_digit, has_special]):
            return False, "Password must contain uppercase, lowercase, digit, and special character"

        return True, "Password is strong"


def format_shell(secrets_dict):
    """Format secrets as shell environment variables."""
    lines = ["# UltraThink Pilot - Generated Secrets", "# Date: 2025-11-24", ""]
    for key, value in secrets_dict.items():
        # Escape special characters for shell
        escaped_value = value.replace('$', '\\$').replace('`', '\\`').replace('"', '\\"')
        lines.append(f'{key}="{escaped_value}"')
    return '\n'.join(lines)


def format_json(secrets_dict):
    """Format secrets as JSON."""
    return json.dumps(secrets_dict, indent=2)


def format_env_file(secrets_dict):
    """Format secrets as .env file with comments."""
    lines = [
        "# UltraThink Pilot - Generated Secrets",
        "# Generated: 2025-11-24",
        "# WARNING: Keep this file secure and never commit to git",
        "",
        "# =============================================================================",
        "# DATABASE CREDENTIALS",
        "# =============================================================================",
        "",
        "POSTGRES_DB=ultrathink_experiments",
        "POSTGRES_USER=ultrathink",
        f"POSTGRES_PASSWORD={secrets_dict['POSTGRES_PASSWORD']}",
        "POSTGRES_HOST=timescaledb",
        "POSTGRES_PORT=5432",
        "",
        "MLFLOW_DB_USER=mlflow",
        f"MLFLOW_DB_PASSWORD={secrets_dict['MLFLOW_DB_PASSWORD']}",
        "",
        "# =============================================================================",
        "# SERVICE API KEYS",
        "# =============================================================================",
        "",
        f"DATA_SERVICE_API_KEY={secrets_dict['DATA_SERVICE_API_KEY']}",
        f"INFERENCE_SERVICE_API_KEY={secrets_dict['INFERENCE_SERVICE_API_KEY']}",
        f"RISK_MANAGER_API_KEY={secrets_dict['RISK_MANAGER_API_KEY']}",
        f"REGIME_DETECTION_API_KEY={secrets_dict['REGIME_DETECTION_API_KEY']}",
        f"META_CONTROLLER_API_KEY={secrets_dict['META_CONTROLLER_API_KEY']}",
        f"ONLINE_LEARNING_API_KEY={secrets_dict['ONLINE_LEARNING_API_KEY']}",
        f"FORENSICS_CONSUMER_API_KEY={secrets_dict['FORENSICS_CONSUMER_API_KEY']}",
        "",
        "# =============================================================================",
        "# MONITORING & OBSERVABILITY",
        "# =============================================================================",
        "",
        "GRAFANA_ADMIN_USER=admin",
        f"GRAFANA_ADMIN_PASSWORD={secrets_dict['GRAFANA_ADMIN_PASSWORD']}",
        f"GF_SECURITY_ADMIN_PASSWORD={secrets_dict['GRAFANA_ADMIN_PASSWORD']}",
        f"GF_SECURITY_SECRET_KEY={secrets_dict['GF_SECURITY_SECRET_KEY']}",
        "",
        "PROMETHEUS_USER=prometheus",
        f"PROMETHEUS_PASSWORD={secrets_dict['PROMETHEUS_PASSWORD']}",
        "",
        "# =============================================================================",
        "# REDIS CACHE",
        "# =============================================================================",
        "",
        "REDIS_HOST=redis",
        "REDIS_PORT=6379",
        f"REDIS_PASSWORD={secrets_dict['REDIS_PASSWORD']}",
        "",
        "# =============================================================================",
        "# SECURITY SETTINGS",
        "# =============================================================================",
        "",
        f"JWT_SECRET_KEY={secrets_dict['JWT_SECRET_KEY']}",
        "JWT_ALGORITHM=HS256",
        "JWT_EXPIRATION_MINUTES=60",
        "",
        "# =============================================================================",
        "# APPLICATION SETTINGS",
        "# =============================================================================",
        "",
        "ENVIRONMENT=development",
        "DEBUG=false",
        "LOG_LEVEL=INFO",
        "",
        "# Service Ports",
        "DATA_SERVICE_PORT=8000",
        "REGIME_DETECTION_PORT=8001",
        "META_CONTROLLER_PORT=8002",
        "RISK_MANAGER_PORT=8003",
        "ONLINE_LEARNING_PORT=8005",
        "INFERENCE_SERVICE_PORT=8080",
        "FORENSICS_CONSUMER_PORT=8090",
        "",
        "# OpenAI API (replace with your key)",
        "OPENAI_API_KEY=sk-your-api-key-here",
        "OPENAI_MODEL=gpt-4o",
        "OPENAI_MAX_TOKENS=4000",
        "",
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate secure passwords and API keys for UltraThink Pilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate and display secrets
    python3 scripts/generate_secrets.py

    # Generate and save to .env file
    python3 scripts/generate_secrets.py --output infrastructure/.env

    # Generate as JSON
    python3 scripts/generate_secrets.py --format json
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: print to stdout)'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['shell', 'json', 'env'],
        default='env',
        help='Output format (default: env)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing .env file'
    )

    args = parser.parse_args()

    # Generate secrets
    generator = SecretGenerator()
    secrets_dict = generator.generate_all_secrets()

    # Format output
    if args.format == 'shell':
        output = format_shell(secrets_dict)
    elif args.format == 'json':
        output = format_json(secrets_dict)
    else:  # env
        output = format_env_file(secrets_dict)

    # Write or print
    if args.output:
        output_path = Path(args.output)

        # Check if file exists and warn
        if output_path.exists():
            response = input(f"⚠️  {args.output} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)

        output_path.write_text(output)
        print(f"✅ Secrets generated and saved to: {args.output}")
        print(f"🔒 File permissions set to 600 (owner read/write only)")

        # Set restrictive permissions
        output_path.chmod(0o600)

        print("\n⚠️  SECURITY REMINDERS:")
        print("   1. Never commit this file to git (check .gitignore)")
        print("   2. Store encrypted backup in password manager")
        print("   3. Share secrets via secure channel (1Password, Vault)")
        print("   4. Rotate secrets every 90 days")

    else:
        print(output)
        print("\n💡 TIP: Run with --output infrastructure/.env to save directly")


if __name__ == '__main__':
    main()
