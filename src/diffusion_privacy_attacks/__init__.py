%%bash
cat << 'EOF' > src/diffusion_privacy_attacks/__init__.py
"""Utilities for running diffusion privacy memorization attacks."""

from .attack import AttackConfig, AttackResult, run_extraction_attack

__all__ = ["AttackConfig", "AttackResult", "run_extraction_attack"]
EOF
