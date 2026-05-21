"""
validators.py — URL input validation and heuristic red flags
"""

import re
from urllib.parse import urlparse


def validate_url(url: str):
    if not url or not url.strip():
        return False, "Please enter a URL."
    url = url.strip()
    if len(url) < 4:
        return False, "URL is too short."
    if len(url) > 30_000:
        return False, "URL exceeds maximum supported length."
    test_url = url if url.startswith(("http://", "https://")) else "http://" + url
    try:
        parsed = urlparse(test_url)
    except Exception:
        return False, "Could not parse this URL."
    if not parsed.netloc:
        return False, "No domain found. Please enter a complete URL."
    if "." not in parsed.netloc and parsed.netloc != "localhost":
        return False, f"'{parsed.netloc}' does not look like a valid domain."
    return True, "OK"


def clean_url(url: str):
    return url.strip()


def heuristic_flags(url: str):
    """Quick rule-based red flags shown alongside the ML prediction."""
    flags = []
    url_lower = url.lower()

    if len(url) > 100:
        flags.append(f"Very long URL ({len(url)} chars) — often used to hide real destination")
    if url.count(".") > 4:
        flags.append(f"Many dots ({url.count('.')} total) — subdomain stacking to impersonate brands")
    if "@" in url:
        flags.append("Contains @ symbol — browser ignores everything before it")
    if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
        flags.append("Domain is a raw IP address — avoids traceable domain registration")

    keywords = ["login", "secure", "verify", "update", "confirm", "banking", "webscr"]
    found = [k for k in keywords if k in url_lower]
    if found:
        flags.append(f"Suspicious keywords found: {', '.join(found)}")

    free = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "website", "space"}
    try:
        tld = urlparse(url if url.startswith("http") else "http://" + url).netloc.split(".")[-1].lower()
        if tld in free:
            flags.append(f"Free TLD '.{tld}' — commonly used for zero-cost phishing domains")
    except Exception:
        pass

    if url.count("-") >= 3:
        flags.append(f"Many hyphens ({url.count('-')}) — e.g. paypal-secure-login.com pattern")
    if re.search(r"%[0-9A-Fa-f]{2}", url):
        flags.append("Percent-encoded characters — used to obscure malicious content")

    return flags
