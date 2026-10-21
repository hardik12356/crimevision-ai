"""
Token Budget & Rate Limiting Manager for Groq API.

Manages token estimation, request budgeting, sliding-window rate tracking,
concurrency locks, and bounded retries to strictly respect Groq provider
limits (e.g., 8,000 Tokens Per Minute for openai/gpt-oss-20b).
"""

import collections
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import tiktoken for accurate BPE token counting
try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


class GroqRateLimitExceeded(Exception):
    """Raised when a request cannot be satisfied within the configured rate budget."""

    def __init__(self, message: str, wait_seconds: float = 0.0, retryable: bool = False):
        super().__init__(message)
        self.wait_seconds = wait_seconds
        self.retryable = retryable


class TokenBudgetManager:
    """
    Thread-safe token estimator, budget enforcer, and sliding-window rate tracker.
    Protects against Groq HTTP 413 / 429 rate limit errors.
    """

    # Default limits (configurable via environment variables)
    DEFAULT_TPM_LIMIT = 8000
    DEFAULT_MAX_INPUT_TOKENS = 4200
    DEFAULT_MAX_OUTPUT_TOKENS_QA = 800
    DEFAULT_MAX_OUTPUT_TOKENS_HYPOTHESIS = 1200
    DEFAULT_MAX_REQUEST_TOTAL = 5200

    def __init__(
        self,
        tpm_limit: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ):
        env_tpm = os.getenv("GROQ_TPM_LIMIT")
        self.tpm_limit = int(env_tpm) if env_tpm else (tpm_limit or self.DEFAULT_TPM_LIMIT)

        env_input = os.getenv("GROQ_MAX_INPUT_TOKENS")
        self.max_input_tokens = int(env_input) if env_input else (max_input_tokens or self.DEFAULT_MAX_INPUT_TOKENS)

        self.max_output_tokens_qa = self.DEFAULT_MAX_OUTPUT_TOKENS_QA
        self.max_output_tokens_hypothesis = self.DEFAULT_MAX_OUTPUT_TOKENS_HYPOTHESIS

        self._lock = threading.Lock()
        # Sliding window of (timestamp_epoch, token_count)
        self._history: collections.deque[tuple[float, int]] = collections.deque()

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Estimate token count accurately using tiktoken if available,
        or a conservative 3.5 chars/token heuristic as fallback.
        """
        if not text:
            return 0
        if _TIKTOKEN_ENCODER is not None:
            try:
                return len(_TIKTOKEN_ENCODER.encode(text))
            except Exception:
                pass
        # Conservative fallback: 3.5 characters per token ensures an upper bound
        return max(1, int(len(text) / 3.5))

    def _prune_expired(self, now: float) -> None:
        """Remove entries older than 60 seconds from the sliding window."""
        cutoff = now - 60.0
        while self._history and self._history[0][0] <= cutoff:
            self._history.popleft()

    def get_current_tpm_usage(self) -> int:
        """Calculate total tokens consumed in the last 60 seconds."""
        now = time.time()
        with self._lock:
            self._prune_expired(now)
            return sum(tokens for _, tokens in self._history)

    def can_proceed(self, estimated_tokens: int) -> tuple[bool, float, str]:
        """
        Check if a request with `estimated_tokens` can proceed under the TPM limit.
        Returns (can_proceed, wait_time_seconds, reason_message).
        """
        now = time.time()
        with self._lock:
            self._prune_expired(now)
            current_usage = sum(tokens for _, tokens in self._history)

            # Check if this single request exceeds total TPM capacity
            if estimated_tokens > self.tpm_limit:
                return (
                    False,
                    0.0,
                    f"Single request of ~{estimated_tokens} tokens exceeds total provider limit of {self.tpm_limit} TPM.",
                )

            # Check if current window has room
            if current_usage + estimated_tokens <= self.tpm_limit:
                return (True, 0.0, "Capacity available")

            # Window is full. Calculate wait time until enough tokens expire
            needed_reduction = (current_usage + estimated_tokens) - self.tpm_limit
            accumulated_reduction = 0
            wait_time = 0.0

            for entry_time, entry_tokens in self._history:
                accumulated_reduction += entry_tokens
                if accumulated_reduction >= needed_reduction:
                    wait_time = max(0.0, (entry_time + 60.0) - now + 0.1)
                    break

            # If wait time is brief (<= 3.0 seconds), allow waiting
            if 0.0 < wait_time <= 3.0:
                return (True, wait_time, f"Brief rate backoff needed ({wait_time:.1f}s)")

            return (
                False,
                wait_time,
                f"Rate limit capacity reached ({current_usage}/{self.tpm_limit} TPM used in last 60s). "
                f"Next slot in {wait_time:.1f}s.",
            )

    def record_usage(self, token_count: int) -> None:
        """Record token consumption in the sliding window."""
        if token_count <= 0:
            return
        now = time.time()
        with self._lock:
            self._prune_expired(now)
            self._history.append((now, token_count))
            logger.debug(
                f"[TokenBudget] Recorded {token_count} tokens. "
                f"Current 60s window: {sum(t for _, t in self._history)}/{self.tpm_limit} TPM"
            )

    def reset(self) -> None:
        """Reset usage history (primarily for tests)."""
        with self._lock:
            self._history.clear()


# Global shared instance
default_token_budget = TokenBudgetManager()
