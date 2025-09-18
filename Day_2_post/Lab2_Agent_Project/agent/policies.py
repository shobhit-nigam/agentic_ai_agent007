import logging, time
from typing import Callable, Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def call_with_policies(tool_name: str, tool_fn: Callable[..., str], args: Dict,
                       *, denylist, max_attempts=2, backoff_sec=0.2, enable_logging=True) -> str:
    """Central helper: guardrails + logging + retry around any tool call."""
    # 1) Guardrails
    text_blob = " ".join(str(v) for v in args.values()).lower()
    if any(bad in text_blob for bad in denylist):
        if enable_logging: log.warning(f"BLOCKED {tool_name}: denylist hit")
        return "Request blocked (guardrails)."

    # 2) Try/Retry loop
    attempts, last_err = 0, None
    while attempts < max_attempts:
        try:
            if enable_logging: log.info(f"CALL {tool_name} args={args}")
            t0 = time.time()
            out = tool_fn(**args)
            dt = int((time.time() - t0) * 1000)
            if enable_logging: log.info(f"OK   {tool_name} in {dt}ms out={out}")
            return out
        except Exception as e:
            attempts += 1
            last_err = e
            if attempts >= max_attempts:
                if enable_logging: log.error(f"FAIL {tool_name} after {attempts}: {e}")
                return f"Sorry — the {tool_name} tool failed after retries."
            if enable_logging: log.warning(f"RETRY {tool_name} attempt {attempts}: {e}")
            time.sleep(backoff_sec * attempts)
