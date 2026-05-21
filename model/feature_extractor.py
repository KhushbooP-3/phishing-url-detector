"""
feature_extractor.py
--------------------
Extracts exactly the 19 features the trained model expects.

These 19 features are what remained after:
  - VarianceThreshold removed: ObfuscationRatio, SpacialCharRatioInURL
  - SmartCorrelatedSelection removed: NoOfLettersInURL, DegitRatioInURL,
    NoOfAmpersandInURL (correlated with others)
  - IsDomainIP excluded (low MI confirmed in EDA)

Feature order must match feature_names.pkl exactly:
  HasObfuscation, URLLength, DomainLength, TLDLength, NoOfSubDomain,
  NoOfDegitsInURL, NoOfQMarkInURL, NoOfEqualsInURL,
  NoOfOtherSpecialCharsInURL, NoOfObfuscatedChar, LetterRatioInURL,
  URLEntropy, IsFreeTLD, BrandSimilarityScore, SuspiciousKeywordFlag,
  NoOfHyphensInDomain, DomainEntropy, HasAtSymbol, TLD_freq
"""

import re
import math
from urllib.parse import urlparse
from difflib import SequenceMatcher

# ── Constants ──────────────────────────────────────────────────────────────────

FREE_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club",
    "online", "site", "website", "space"
}

BRANDS = [
    "google", "paypal", "microsoft", "apple", "amazon", "facebook",
    "netflix", "instagram", "twitter", "linkedin", "yahoo", "ebay",
    "chase", "bankofamerica", "wellsfargo", "dropbox", "adobe", "spotify"
]

SUSPICIOUS_KEYWORDS = [
    "login", "secure", "update", "verify", "confirm",
    "account", "banking", "webscr", "signin", "password",
    "credential", "authenticate", "validation"
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _entropy(text):
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return round(-sum((c/n) * math.log2(c/n) for c in freq.values()), 6)


def _brand_similarity(domain):
    clean = domain.lower().replace("-", "").replace(".", "")
    return round(max(
        SequenceMatcher(None, clean, b).ratio() for b in BRANDS
    ), 6)


def _count_obfuscated(url):
    return len(re.findall(r"%[0-9A-Fa-f]{2}", url))


# ── Main ───────────────────────────────────────────────────────────────────────

def extract_features(url: str, tld_freq: dict) -> dict:
    """
    Extract all 19 features from a URL string.
    tld_freq dict is loaded from tld_freq.pkl.

    Returns dict with keys matching feature_names.pkl order.
    """
    # Parse URL
    test_url = url if url.startswith(("http://", "https://")) else "http://" + url
    try:
        parsed = urlparse(test_url)
    except Exception:
        parsed = urlparse("http://invalid.com")

    domain     = parsed.netloc or ""
    tld_raw    = domain.split(".")[-1].lower() if "." in domain else ""
    subdomains = domain.split(".")[:-2] if domain.count(".") >= 2 else []
    url_len    = len(url)

    # Character counts
    digits      = len(re.findall(r"[0-9]", url))
    letters     = len(re.findall(r"[a-zA-Z]", url))
    obf_chars   = _count_obfuscated(url)
    q_marks     = url.count("?")
    equals      = url.count("=")

    # Other special chars = all non-alphanumeric minus q_marks, equals, ampersands
    url_no_obf   = re.sub(r"%[0-9A-Fa-f]{2}", "", url)
    all_special  = len(re.findall(r"[^a-zA-Z0-9]", url_no_obf))
    ampersands   = url.count("&")
    other_special = max(all_special - q_marks - equals - ampersands, 0)

    features = {
        # Binary
        "HasObfuscation":            int(obf_chars > 0),

        # Count features
        "URLLength":                 url_len,
        "DomainLength":              len(domain),
        "TLDLength":                 len(tld_raw),
        "NoOfSubDomain":             len(subdomains),
        "NoOfDegitsInURL":           digits,
        "NoOfQMarkInURL":            q_marks,
        "NoOfEqualsInURL":           equals,
        "NoOfOtherSpecialCharsInURL": other_special,
        "NoOfObfuscatedChar":        obf_chars,

        # Ratio
        "LetterRatioInURL":          round(letters / url_len, 6) if url_len > 0 else 0.0,

        # Engineered features
        "URLEntropy":                _entropy(url),
        "IsFreeTLD":                 int(tld_raw in FREE_TLDS),
        "BrandSimilarityScore":      _brand_similarity(domain),
        "SuspiciousKeywordFlag":     int(any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS)),
        "NoOfHyphensInDomain":       domain.count("-"),
        "DomainEntropy":             _entropy(domain),
        "HasAtSymbol":               int("@" in url),

        # TLD frequency encoding
        "TLD_freq":                  tld_freq.get(tld_raw, 0.0),
    }

    return features


def get_feature_explanations():
    return {
        "HasObfuscation":             "Contains percent-encoded characters (%20 etc.) to hide content",
        "URLLength":                  "Total URL length — very long URLs are almost always phishing",
        "DomainLength":               "Length of the domain name",
        "TLDLength":                  "Length of the top-level domain (.com = 3)",
        "NoOfSubDomain":              "Number of subdomains — phishing stacks many to impersonate brands",
        "NoOfDegitsInURL":            "Count of digits — phishing URLs embed more numbers",
        "NoOfQMarkInURL":             "Number of ? characters (query parameters)",
        "NoOfEqualsInURL":            "Number of = characters (key=value pairs)",
        "NoOfOtherSpecialCharsInURL": "Count of other special characters",
        "NoOfObfuscatedChar":         "Number of percent-encoded characters",
        "LetterRatioInURL":           "Proportion of alphabetic characters in URL",
        "URLEntropy":                 "Randomness of URL — auto-generated phishing URLs score high",
        "IsFreeTLD":                  "Uses a free TLD (.tk .ml .xyz) — easy to abuse for phishing",
        "BrandSimilarityScore":       "How similar domain is to a known brand — typosquatting signal",
        "SuspiciousKeywordFlag":      "Contains words like login, secure, verify, confirm",
        "NoOfHyphensInDomain":        "Hyphens in domain — e.g. paypal-secure-login.com pattern",
        "DomainEntropy":              "Randomness of domain name — random strings = phishing generators",
        "HasAtSymbol":                "Contains @ — browser ignores everything before it",
        "TLD_freq":                   "How common this TLD is — rare TLDs more often phishing",
    }
