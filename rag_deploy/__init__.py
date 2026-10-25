"""Production admissions application."""
import sys

# Use Windows' certificate validation for locally trusted HTTPS proxies as well
# as public CAs. This must run before the API SDKs initialize their transports.
if sys.platform == "win32":
    import truststore

    truststore.inject_into_ssl()
