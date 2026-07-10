"""
URL feature extractor - PhishGuard AI backend.

Computes the 20 URL-derivable PhiUSIIL features for a raw URL string,
matching the definitions used to train the deployment model. Lookup
tables (TLD legitimacy probability, character probability) are learned
from the training corpus and shipped in ../outputs.
"""

import ipaddress
import os
import re
from urllib.parse import urlparse

import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS = os.path.join(_HERE, "..", "..", "outputs")

TLD_LOOKUP = joblib.load(os.path.join(_OUTPUTS, "tld_probability_lookup.joblib"))
CHAR_LOOKUP = joblib.load(os.path.join(_OUTPUTS, "char_probability_lookup.joblib"))
FEATURE_ORDER = joblib.load(os.path.join(_OUTPUTS, "deployment_features.joblib"))

_OBFUSCATED = re.compile(r"%[0-9a-fA-F]{2}|@|0x[0-9a-fA-F]+")


def _is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host.split(":")[0])
        return 1
    except ValueError:
        return 0


def extract_features(url: str) -> dict:
    """Return a dict of the 20 URL-only features, keyed by PhiUSIIL name."""
    raw = url.strip()
    if not raw.lower().startswith(("http://", "https://")):
        raw = "http://" + raw

    parsed = urlparse(raw)
    domain = parsed.netloc.lower()
    host = domain.split(":")[0]

    url_len = len(raw)
    letters = re.findall(r"[a-zA-Z]", raw)
    digits = re.findall(r"[0-9]", raw)
    specials = re.findall(r"[^a-zA-Z0-9]", raw)
    # "Other" special chars exclude the structural ones counted separately
    other_specials = [
        c for c in specials if c not in ("=", "?", "&", "/", ":", ".")
    ]

    parts = host.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    # Subdomain count: labels before the registrable domain (approximation)
    n_sub = max(len(parts) - 2, 0)

    obfuscated = _OBFUSCATED.findall(raw)
    n_obf = len(obfuscated)

    char_probs = [CHAR_LOOKUP.get(c, 1e-6) for c in raw.lower()]

    feats = {
        "URLLength": url_len,
        "DomainLength": len(domain),
        "IsDomainIP": _is_ip(host),
        "TLDLegitimateProb": float(TLD_LOOKUP.get(tld, 0.05)),
        "URLCharProb": float(np.mean(char_probs)) if char_probs else 0.0,
        "TLDLength": len(tld),
        "NoOfSubDomain": n_sub,
        "HasObfuscation": int(n_obf > 0),
        "NoOfObfuscatedChar": n_obf,
        "ObfuscationRatio": n_obf / max(url_len, 1),
        "NoOfLettersInURL": len(letters),
        "LetterRatioInURL": round(len(letters) / max(url_len, 1), 3),
        "NoOfDegitsInURL": len(digits),
        "DegitRatioInURL": round(len(digits) / max(url_len, 1), 3),
        "NoOfEqualsInURL": raw.count("="),
        "NoOfQMarkInURL": raw.count("?"),
        "NoOfAmpersandInURL": raw.count("&"),
        "NoOfOtherSpecialCharsInURL": len(other_specials),
        "SpacialCharRatioInURL": round(len(other_specials) / max(url_len, 1), 3),
        "IsHTTPS": int(parsed.scheme == "https"),
    }
    return feats


def feature_vector(url: str) -> np.ndarray:
    """Features as a (1, n) array in the exact order the model expects."""
    feats = extract_features(url)
    return np.array([[feats[name] for name in FEATURE_ORDER]])
