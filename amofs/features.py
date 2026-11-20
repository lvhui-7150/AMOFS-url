"""URL feature extraction with an explicit per-feature modification-cost tier.

Every feature is declared once in ``FEATURE_SPEC`` together with the cost tier
used by the threat model (Section 3 of the manuscript):

    cheap     (d = 1): attacker-controlled string content (path, query, URL
                       length, character statistics).  Changing it costs
                       nothing beyond editing the link.
    moderate  (d = 3): host *configuration* under the attacker's control but
                       requiring infrastructure work (sub-domain layout, TLS,
                       ports, www prefix, host-name composition).
    expensive (d = 8): bound to domain registration / TLD choice / hosting
                       identity (registered-domain composition, TLD class,
                       IP-literal host, punycode, shortener or free-hosting
                       provider).  Changing these means acquiring a new
                       registered domain or changing provider.

The extractor is deterministic, offline, and needs only the raw URL string, so
it can be applied identically to every public corpus.  ``tldextract`` is used
for the registered-domain split when installed; a built-in public-suffix
fallback is used otherwise.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qsl, unquote, urlparse

try:  # optional dependency
    import tldextract  # type: ignore

    _TLD = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)
except Exception:  # pragma: no cover - fallback path
    _TLD = None

COST_CHEAP = 1.0
COST_MODERATE = 3.0
COST_EXPENSIVE = 8.0
TIER_COST = {"cheap": COST_CHEAP, "moderate": COST_MODERATE, "expensive": COST_EXPENSIVE}

SUSPICIOUS_TOKENS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "update",
    "secure", "security", "account", "bank", "free", "bonus", "paypal",
    "wp-admin", "wp-content", "wp-includes", "download", "invoice", "confirm",
    "password", "wallet", "crypto", "gift", "prize", "support", "billing",
    "recover", "unlock", "suspend", "limited", "alert", "webscr", "cmd",
)
BRAND_TOKENS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "facebook", "instagram", "whatsapp", "amazon", "netflix", "dhl", "fedex",
    "usps", "ups", "chase", "wellsfargo", "bankofamerica", "hsbc", "santander",
    "barclays", "ing", "bbva", "dropbox", "docusign", "linkedin", "twitter",
    "steam", "binance", "coinbase", "metamask", "att", "verizon", "orange",
)
SHORTENER_DOMAINS = (
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "bitly.com", "tiny.cc", "shorturl.at", "rb.gy",
    "lnkd.in", "s.id", "t.ly", "v.gd", "clck.ru", "qr.ae", "adf.ly",
)
FREE_HOSTING_SUFFIXES = (
    "000webhostapp.com", "github.io", "gitlab.io", "netlify.app", "vercel.app",
    "weebly.com", "wixsite.com", "blogspot.com", "wordpress.com", "glitch.me",
    "herokuapp.com", "firebaseapp.com", "web.app", "pages.dev", "repl.co",
    "duckdns.org", "no-ip.org", "ddns.net", "hopto.org", "zapto.org",
    "square.site", "godaddysites.com", "yolasite.com", "webnode.page",
    "myfreesites.net", "sites.google.com", "freehostia.com", "azurewebsites.net",
    "cloudfront.net", "amazonaws.com", "r2.dev", "workers.dev", "surge.sh",
)
SUSPICIOUS_TLDS = (
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "club", "work", "click",
    "link", "info", "online", "site", "website", "space", "icu", "cyou", "rest",
    "monster", "cam", "quest", "sbs", "bond", "lol", "live", "fit", "zip", "mov",
)
EXECUTABLE_SUFFIXES = (".exe", ".scr", ".dll", ".bat", ".cmd", ".js", ".jar",
                       ".apk", ".msi", ".bin", ".sh", ".vbs", ".ps1", ".hta",
                       ".com", ".pif", ".elf", ".dmg", ".iso", ".img")
ARCHIVE_SUFFIXES = (".zip", ".rar", ".7z", ".gz", ".tar", ".bz2", ".xz", ".cab",
                    ".arj", ".lzh", ".ace")
DOCUMENT_SUFFIXES = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                     ".rtf", ".odt", ".one", ".lnk")

# Two-label public suffixes used when tldextract is unavailable.
_FALLBACK_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "ltd.uk", "plc.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "com.br", "net.br", "org.br", "gov.br",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp",
    "com.cn", "net.cn", "org.cn", "edu.cn", "gov.cn",
    "co.in", "net.in", "org.in", "ac.in", "gov.in",
    "com.tr", "com.mx", "com.ar", "co.za", "com.sg", "co.kr", "co.nz",
    "com.hk", "com.tw", "com.ua", "com.pl", "com.vn", "com.eg", "com.ng",
    "com.pk", "com.my", "com.ph", "com.sa", "co.id", "or.id", "ac.id",
    "com.co", "com.pe", "com.ve", "com.ec", "com.uy", "com.py", "com.bo",
    "com.do", "com.gt", "com.sv", "com.hn", "com.ni", "com.pa", "com.cu",
    "com.pr", "co.il", "org.il", "com.bd", "com.np", "com.kw", "com.qa",
    "com.lb", "com.jo", "com.om", "com.bh", "co.ke", "co.tz", "co.ug",
    "com.gh", "com.et", "com.ru", "spb.ru", "msk.ru", "com.by", "com.kz",
}

# ---------------------------------------------------------------------------
# Feature declaration: (name, tier).  Order is the column order of the CSV.
# ---------------------------------------------------------------------------
FEATURE_SPEC: List[Tuple[str, str]] = [
    # ---- cheap: whole-URL string statistics -------------------------------
    ("url_len", "cheap"),
    ("url_entropy", "cheap"),
    ("url_digit_ratio", "cheap"),
    ("url_alpha_ratio", "cheap"),
    ("url_special_ratio", "cheap"),
    ("url_upper_ratio", "cheap"),
    ("dot_count", "cheap"),
    ("slash_count", "cheap"),
    ("hyphen_count", "cheap"),
    ("underscore_count", "cheap"),
    ("at_count", "cheap"),
    ("question_count", "cheap"),
    ("equal_count", "cheap"),
    ("amp_count", "cheap"),
    ("percent_count", "cheap"),
    ("tilde_count", "cheap"),
    ("plus_count", "cheap"),
    ("encoded_char_count", "cheap"),
    ("double_slash_in_path", "cheap"),
    ("repeated_char_run", "cheap"),
    # ---- cheap: path / query / fragment --------------------------------
    ("path_len", "cheap"),
    ("path_entropy", "cheap"),
    ("path_depth", "cheap"),
    ("path_digit_ratio", "cheap"),
    ("path_token_count", "cheap"),
    ("longest_path_token_len", "cheap"),
    ("avg_path_token_len", "cheap"),
    ("file_ext_len", "cheap"),
    ("executable_suffix", "cheap"),
    ("archive_suffix", "cheap"),
    ("document_suffix", "cheap"),
    ("query_len", "cheap"),
    ("query_entropy", "cheap"),
    ("query_param_count", "cheap"),
    ("query_has_url", "cheap"),
    ("fragment_len", "cheap"),
    ("suspicious_token_count", "cheap"),
    ("brand_token_in_path", "cheap"),
    ("digit_token_count", "cheap"),
    ("hex_token_count", "cheap"),
    # ---- moderate: host configuration ----------------------------------
    ("has_https", "moderate"),
    ("has_port", "moderate"),
    ("nonstandard_port", "moderate"),
    ("www_prefix", "moderate"),
    ("host_len", "moderate"),
    ("host_entropy", "moderate"),
    ("host_dot_count", "moderate"),
    ("host_hyphen_count", "moderate"),
    ("host_digit_ratio", "moderate"),
    ("subdomain_count", "moderate"),
    ("subdomain_len", "moderate"),
    ("subdomain_entropy", "moderate"),
    ("longest_host_token_len", "moderate"),
    ("brand_token_in_subdomain", "moderate"),
    ("suspicious_token_in_host", "moderate"),
    # ---- expensive: registration / TLD / provider identity -------------
    ("has_ip_host", "expensive"),
    ("punycode_host", "expensive"),
    ("shortener_domain", "expensive"),
    ("free_hosting_domain", "expensive"),
    ("tld_len", "expensive"),
    ("tld_is_suspicious", "expensive"),
    ("tld_is_cctld", "expensive"),
    ("registered_domain_len", "expensive"),
    ("registered_domain_entropy", "expensive"),
    ("registered_domain_digit_ratio", "expensive"),
    ("registered_domain_hyphen_count", "expensive"),
    ("brand_token_in_registered_domain", "expensive"),
]

FEATURE_NAMES: List[str] = [name for name, _ in FEATURE_SPEC]
FEATURE_TIERS: Dict[str, str] = dict(FEATURE_SPEC)
FEATURE_COSTS: Dict[str, float] = {n: TIER_COST[t] for n, t in FEATURE_SPEC}


def feature_names() -> List[str]:
    return list(FEATURE_NAMES)


def feature_costs(names: Iterable[str] | None = None) -> List[float]:
    """Cost vector d_i for the given feature names (default: all features)."""
    names = list(names) if names is not None else FEATURE_NAMES
    out = []
    for n in names:
        if n not in FEATURE_COSTS:
            raise KeyError(f"feature '{n}' has no declared cost tier")
        out.append(FEATURE_COSTS[n])
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _normalise_url(raw_url: str) -> str:
    url = str(raw_url).strip().strip('"').strip("'")
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
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _port(parsed) -> int | None:
    try:
        return parsed.port
    except ValueError:
        return None


def _longest_run(text: str) -> int:
    if not text:
        return 0
    longest = cur = 1
    for a, b in zip(text, text[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)
    return longest


_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.split(text) if t]


def _split_host(host: str) -> Tuple[str, str, str]:
    """Return (subdomain, registered_domain_label, suffix) for a host name."""
    if not host or _is_ip(host):
        return "", host, ""
    if _TLD is not None:
        ext = _TLD(host)
        return ext.subdomain, ext.domain, ext.suffix
    labels = host.split(".")
    if len(labels) < 2:
        return "", host, ""
    last2 = ".".join(labels[-2:])
    if last2 in _FALLBACK_SUFFIXES and len(labels) >= 3:
        suffix = last2
        domain = labels[-3]
        sub = ".".join(labels[:-3])
    else:
        suffix = labels[-1]
        domain = labels[-2]
        sub = ".".join(labels[:-2])
    return sub, domain, suffix


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# main extractor
# ---------------------------------------------------------------------------
def extract_url_features(raw_url: str) -> Dict[str, float]:
    url = _normalise_url(raw_url)
    parsed = urlparse(url)
    decoded = unquote(url)
    lower_decoded = decoded.lower()

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    is_ip = _is_ip(host)
    sub, dom, suffix = _split_host(host)
    tld = suffix.rsplit(".", 1)[-1] if suffix else ""
    registered = f"{dom}.{suffix}" if suffix else dom
    host_labels = [p for p in host.split(".") if p]
    sub_labels = [p for p in sub.split(".") if p]

    path_tokens = _tokens(path)
    path_token_lens = [len(t) for t in path_tokens]
    all_tokens = _tokens(host + " " + path + " " + query)
    host_tokens = _tokens(host)

    n = len(url)
    digits = sum(ch.isdigit() for ch in url)
    alphas = sum(ch.isalpha() for ch in url)
    uppers = sum(ch.isupper() for ch in url)
    specials = sum(not ch.isalnum() for ch in url)

    last_seg = path.rsplit("/", 1)[-1]
    ext = last_seg.rsplit(".", 1)[-1].lower() if "." in last_seg else ""
    lower_path = path.lower()
    port = _port(parsed)
    default_port = {"http": 80, "https": 443}.get(parsed.scheme.lower())

    params = parse_qsl(query, keep_blank_values=True)
    query_has_url = int(bool(re.search(r"(https?:|%3a%2f%2f|www\.)", query.lower())))

    hex_tokens = sum(1 for t in all_tokens if len(t) >= 8 and re.fullmatch(r"[0-9a-fA-F]+", t))
    digit_tokens = sum(1 for t in all_tokens if any(c.isdigit() for c in t))

    path_query_lower = (path + "?" + query).lower()
    suspicious_in_url = sum(1 for tok in SUSPICIOUS_TOKENS if tok in lower_decoded)
    suspicious_in_host = int(any(tok in host for tok in SUSPICIOUS_TOKENS))
    brand_in_path = int(any(b in path_query_lower for b in BRAND_TOKENS))
    brand_in_sub = int(any(b in sub for b in BRAND_TOKENS)) if sub else 0
    brand_in_dom = int(any(b in dom for b in BRAND_TOKENS)) if dom else 0

    shortener = int(registered in SHORTENER_DOMAINS or host in SHORTENER_DOMAINS)
    free_host = int(any(host == s or host.endswith("." + s) for s in FREE_HOSTING_SUFFIXES))

    feats: Dict[str, float] = {
        # cheap, whole URL
        "url_len": float(n),
        "url_entropy": _entropy(url),
        "url_digit_ratio": _ratio(digits, n),
        "url_alpha_ratio": _ratio(alphas, n),
        "url_special_ratio": _ratio(specials, n),
        "url_upper_ratio": _ratio(uppers, n),
        "dot_count": float(url.count(".")),
        "slash_count": float(url.count("/")),
        "hyphen_count": float(url.count("-")),
        "underscore_count": float(url.count("_")),
        "at_count": float(url.count("@")),
        "question_count": float(url.count("?")),
        "equal_count": float(url.count("=")),
        "amp_count": float(url.count("&")),
        "percent_count": float(url.count("%")),
        "tilde_count": float(url.count("~")),
        "plus_count": float(url.count("+")),
        "encoded_char_count": float(len(re.findall(r"%[0-9a-fA-F]{2}", url))),
        "double_slash_in_path": float("//" in path),
        "repeated_char_run": float(_longest_run(url)),
        # cheap, path/query/fragment
        "path_len": float(len(path)),
        "path_entropy": _entropy(path),
        "path_depth": float(len([p for p in path.split("/") if p])),
        "path_digit_ratio": _ratio(sum(c.isdigit() for c in path), len(path)),
        "path_token_count": float(len(path_tokens)),
        "longest_path_token_len": float(max(path_token_lens) if path_token_lens else 0),
        "avg_path_token_len": float(sum(path_token_lens) / len(path_token_lens)) if path_token_lens else 0.0,
        "file_ext_len": float(len(ext) if 0 < len(ext) <= 6 else 0),
        "executable_suffix": float(any(lower_path.endswith(s) for s in EXECUTABLE_SUFFIXES)),
        "archive_suffix": float(any(lower_path.endswith(s) for s in ARCHIVE_SUFFIXES)),
        "document_suffix": float(any(lower_path.endswith(s) for s in DOCUMENT_SUFFIXES)),
        "query_len": float(len(query)),
        "query_entropy": _entropy(query),
        "query_param_count": float(len(params)),
        "query_has_url": float(query_has_url),
        "fragment_len": float(len(fragment)),
        "suspicious_token_count": float(suspicious_in_url),
        "brand_token_in_path": float(brand_in_path),
        "digit_token_count": float(digit_tokens),
        "hex_token_count": float(hex_tokens),
        # moderate, host configuration
        "has_https": float(parsed.scheme.lower() == "https"),
        "has_port": float(port is not None),
        "nonstandard_port": float(port is not None and port != default_port),
        "www_prefix": float(host.startswith("www.")),
        "host_len": float(len(host)),
        "host_entropy": _entropy(host),
        "host_dot_count": float(host.count(".")),
        "host_hyphen_count": float(host.count("-")),
        "host_digit_ratio": _ratio(sum(c.isdigit() for c in host), len(host)),
        "subdomain_count": float(0 if is_ip else len(sub_labels)),
        "subdomain_len": float(len(sub)),
        "subdomain_entropy": _entropy(sub),
        "longest_host_token_len": float(max((len(t) for t in host_tokens), default=0)),
        "brand_token_in_subdomain": float(brand_in_sub),
        "suspicious_token_in_host": float(suspicious_in_host),
        # expensive, registration / TLD / provider identity
        "has_ip_host": float(is_ip),
        "punycode_host": float("xn--" in host),
        "shortener_domain": float(shortener),
        "free_hosting_domain": float(free_host),
        "tld_len": float(len(tld)),
        "tld_is_suspicious": float(tld in SUSPICIOUS_TLDS),
        "tld_is_cctld": float(len(tld) == 2),
        "registered_domain_len": float(len(dom)),
        "registered_domain_entropy": _entropy(dom),
        "registered_domain_digit_ratio": _ratio(sum(c.isdigit() for c in dom), len(dom)),
        "registered_domain_hyphen_count": float(dom.count("-")),
        "brand_token_in_registered_domain": float(brand_in_dom),
    }
    # guarantee column order and completeness
    return {name: float(feats[name]) for name in FEATURE_NAMES}


def extract_many(urls: Iterable[str]):
    """Yield feature dicts for an iterable of URLs (robust to malformed input)."""
    for u in urls:
        try:
            yield extract_url_features(u)
        except Exception:
            yield {name: 0.0 for name in FEATURE_NAMES}


if __name__ == "__main__":  # quick self-check
    demo = extract_url_features("http://paypal-login.secure-update.tk:8080/a/b/verify.php?id=1&r=http://x.y")
    assert len(demo) == len(FEATURE_SPEC) == len(set(FEATURE_NAMES))
    for k, v in demo.items():
        print(f"{k:36s} {FEATURE_TIERS[k]:9s} {v:.3f}")
    print(f"{len(demo)} features: "
          f"{sum(t == 'cheap' for t in FEATURE_TIERS.values())} cheap, "
          f"{sum(t == 'moderate' for t in FEATURE_TIERS.values())} moderate, "
          f"{sum(t == 'expensive' for t in FEATURE_TIERS.values())} expensive")
