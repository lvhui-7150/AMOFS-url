"""URL feature extraction used by the public-data preparation script.

The extractor is intentionally self-contained and deterministic: it only uses
lexical and URL-structure signals that are available for every raw URL. Host
reputation or WHOIS features can be added later when a local enrichment source
is available, but these features are enough to run the AMOFS pipeline on public
raw URL corpora without external API keys.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Dict, Iterable, List
from urllib.parse import parse_qsl, unquote, urlparse


SUSPICIOUS_TOKENS = (
    "login", "verify", "update", "secure", "account", "bank", "free",
    "bonus", "paypal", "signin", "wp-admin", "download", "invoice",
    "confirm", "password", "wallet", "crypto", "gift", "prize",
)
SHORTENER_TOKENS = (
    "bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "bitly.com",
)
EXECUTABLE_SUFFIXES = (
    ".exe", ".scr", ".dll", ".bat", ".cmd", ".js", ".jar", ".zip",
    ".rar", ".7z", ".apk", ".msi", ".bin", ".sh",
)


def _normalise_url(raw_url: str) -> str:
    url = str(raw_url).strip()
    if not url:
        return "http://"
    if "://" not in url:
        url = "http://" + url
    return url


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _is_ip_address(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return 0


def _has_valid_port(parsed) -> int:
    try:
        return int(parsed.port is not None)
    except ValueError:
        return 0


def _longest_run(text: str) -> int:
    if not text:
        return 0
    longest = current = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            prev = ch
    return longest


def _tokens(parts: Iterable[str]) -> List[str]:
    joined = " ".join(part for part in parts if part)
    return [token for token in re.split(r"[^A-Za-z0-9]+", joined) if token]


def extract_url_features(raw_url: str) -> Dict[str, float]:
    """Return deterministic numeric features for one URL."""
    url = _normalise_url(raw_url)
    parsed = urlparse(url)
    decoded_url = unquote(url)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    host_parts = [part for part in host.split(".") if part]
    subdomain_count = max(0, len(host_parts) - 2) if not _is_ip_address(host) else 0
    all_tokens = _tokens([host, path, query])
    token_lengths = [len(token) for token in all_tokens]

    length = len(url)
    alpha = sum(ch.isalpha() for ch in url)
    digit = sum(ch.isdigit() for ch in url)
    special = sum(not ch.isalnum() for ch in url)
    hex_tokens = sum(1 for token in all_tokens if len(token) >= 8 and re.fullmatch(r"[0-9a-fA-F]+", token))
    digit_tokens = sum(1 for token in all_tokens if any(ch.isdigit() for ch in token))
    params = parse_qsl(query, keep_blank_values=True)

    lower_url = decoded_url.lower()
    suspicious_count = sum(1 for token in SUSPICIOUS_TOKENS if token in lower_url)
    shortener = int(any(token in host for token in SHORTENER_TOKENS))
    executable = int(any(path.lower().endswith(suffix) for suffix in EXECUTABLE_SUFFIXES))

    return {
        "url_len": float(length),
        "url_entropy": _entropy(url),
        "url_digit_ratio": digit / max(length, 1),
        "url_alpha_ratio": alpha / max(length, 1),
        "url_special_count": float(special),
        "dot_count": float(url.count(".")),
        "slash_count": float(url.count("/")),
        "hyphen_count": float(url.count("-")),
        "at_count": float(url.count("@")),
        "question_count": float(url.count("?")),
        "equal_count": float(url.count("=")),
        "amp_count": float(url.count("&")),
        "percent_count": float(url.count("%")),
        "encoded_char_count": float(len(re.findall(r"%[0-9a-fA-F]{2}", url))),
        "host_len": float(len(host)),
        "host_entropy": _entropy(host),
        "host_dot_count": float(host.count(".")),
        "subdomain_count": float(subdomain_count),
        "tld_len": float(len(tld)),
        "domain_token_count": float(len(host_parts)),
        "has_ip_host": float(_is_ip_address(host)),
        "has_https": float(parsed.scheme.lower() == "https"),
        "has_port": float(_has_valid_port(parsed) if host else 0),
        "punycode_host": float("xn--" in host),
        "www_prefix": float(host.startswith("www.")),
        "path_len": float(len(path)),
        "path_entropy": _entropy(path),
        "path_depth": float(len([part for part in path.split("/") if part])),
        "query_len": float(len(query)),
        "query_param_count": float(len(params)),
        "suspicious_token_count": float(suspicious_count),
        "shortener_token": float(shortener),
        "executable_suffix": float(executable),
        "longest_token_len": float(max(token_lengths) if token_lengths else 0),
        "avg_token_len": float(sum(token_lengths) / len(token_lengths) if token_lengths else 0),
        "repeated_char_run": float(_longest_run(url)),
        "digit_token_count": float(digit_tokens),
        "hexadecimal_token_count": float(hex_tokens),
    }


def feature_names() -> List[str]:
    """Return feature names in stable extraction order."""
    return list(extract_url_features("http://example.com/path?a=1").keys())
