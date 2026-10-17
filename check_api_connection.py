#!/usr/bin/env python3
"""Diagnose Python HTTPS access. No API keys or model inference requests are sent."""
import json
import os
import platform
import re
import socket
import ssl
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, getproxies, urlopen

ENDPOINTS = {
    'jev': 'https://api.typesafe.ai/v1/systemone',
    'openai': 'https://api.openai.com/v1/models',
}


def safe_text(value):
    text = str(value)
    for name in ('OPENAI_API_KEY', 'TYPESAFE_API_KEY', 'JEV_API_KEY'):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, '[redacted]')
    text = re.sub(r'(?i)(https?://)[^/\s@]+@', r'\1[redacted]@', text)
    return text[:1200]


def error_info(exc):
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        category = 'certificate_verification'
    elif isinstance(reason, socket.gaierror):
        category = 'dns_resolution'
    elif isinstance(reason, (TimeoutError, socket.timeout)):
        category = 'timeout'
    elif isinstance(reason, ConnectionRefusedError):
        category = 'connection_refused'
    else:
        category = 'other_connection_error'
    return {'category': category, 'exception': type(exc).__name__,
            'reason_type': type(reason).__name__, 'reason': safe_text(reason)}


def probe(url, context):
    try:
        request = Request(url, method='GET', headers={'User-Agent': 'routing-connectivity-check/1.0'})
        with urlopen(request, timeout=15, context=context) as response:
            return {'http_response_received': True, 'http_status': response.status}
    except HTTPError as exc:
        # 401/403/405 can be expected without credentials or for a GET-only probe.
        # A responding proxy may also produce an HTTP error; this does not verify API access.
        return {'http_response_received': True, 'http_status': exc.code,
                'note': 'HTTP error response received; no authenticated inference was attempted.'}
    except Exception as exc:
        return {'http_response_received': False, **error_info(exc)}


def main():
    report = {'python': platform.python_version(), 'executable': sys.executable,
              'openssl': ssl.OPENSSL_VERSION,
              'proxy_types_configured': sorted(getproxies()),
              'certificate_env_names_set': [k for k in ('SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE') if os.environ.get(k)],
              'default_certificate_paths': ssl.get_default_verify_paths()._asdict(),
              'note': 'Unauthenticated GET checks only; no API keys sent, no model calls.'}
    try:
        context = ssl.create_default_context()
        report['default_trust'] = {p: probe(u, context) for p, u in ENDPOINTS.items()}
    except Exception as exc:
        report['default_trust_setup_error'] = error_info(exc)
    # Optional comparison only if certifi is already installed and the default check failed.
    failed = any(not r['http_response_received'] for r in report.get('default_trust', {}).values())
    if failed:
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
            report['certifi_trust'] = {p: probe(u, context) for p, u in ENDPOINTS.items()}
        except ImportError:
            report['certifi_trust'] = 'certifi is not installed; optional comparison skipped.'
        except Exception as exc:
            report['certifi_trust_setup_error'] = error_info(exc)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
