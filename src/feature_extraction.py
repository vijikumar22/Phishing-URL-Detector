import re
import urllib.parse as urlparse
#from src.feature_extraction import build_feature_dataframe
import pandas as pd
import tldextract


def has_ip_address(url: str) -> int:
    # Check if the domain is an IP address
    match = re.search(r'(\d{1,3}\.){3}\d{1,3}', url)
    return 1 if match else 0


def count_dots(url: str) -> int:
    return url.count('.')


def count_hyphens(url: str) -> int:
    return url.count('-')


def url_length(url: str) -> int:
    return len(url)


def has_at_symbol(url: str) -> int:
    return 1 if '@' in url else 0


def has_https(url: str) -> int:
    return 1 if url.lower().startswith("https") else 0


def get_num_parameters(url: str) -> int:
    parsed = urlparse.urlparse(url)
    query = parsed.query
    if not query:
        return 0
    return query.count('&') + 1


def get_num_subdirectories(url: str) -> int:
    parsed = urlparse.urlparse(url)
    path = parsed.path
    if not path or path == '/':
        return 0
    return path.count('/')


def suspicious_words(url: str) -> int:
    words = ['secure', 'account', 'update', 'login', 'verify', 'bank', 'confirm', 'payment', 'signin']
    url_lower = url.lower()
    return sum(1 for w in words if w in url_lower)


def get_tld_length(url: str) -> int:
    extracted = tldextract.extract(url)
    tld = extracted.suffix  # com, in, co.in, etc.
    return len(tld) if tld else 0


def get_domain_length(url: str) -> int:
    extracted = tldextract.extract(url)
    domain = extracted.domain
    return len(domain) if domain else 0


def extract_features_from_url(url: str) -> dict:
    """Return a dict of features for a single URL."""
    return {
        "url_length": url_length(url),
        "has_ip": has_ip_address(url),
        "count_dots": count_dots(url),
        "count_hyphens": count_hyphens(url),
        "has_at": has_at_symbol(url),
        "has_https": has_https(url),
        "num_parameters": get_num_parameters(url),
        "num_subdirectories": get_num_subdirectories(url),
        "suspicious_words": suspicious_words(url),
        "tld_length": get_tld_length(url),
        "domain_length": get_domain_length(url),
    }


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must have a column 'url'.
    Returns new DataFrame with extracted features.
    """
    feature_rows = []
    for url in df['url']:
        feature_rows.append(extract_features_from_url(url))

    feature_df = pd.DataFrame(feature_rows)
    return feature_df
