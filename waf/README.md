# WAF Testbed (Skeleton)

- Services:
  - `nginx-modsec`: Nginx + ModSecurity + OWASP CRS (image: `owasp/modsecurity-crs:nginx`).
  - `dvwa`: Vulnerable web app for testing (`vulnerables/web-dvwa`).

- Mounts:
  - `modsecurity.conf`, `crs-setup.conf`, custom patches under `waf/patches/`.
  - Logs under `waf/logs/`.

- Default port: `http://localhost:8080/`.

Notes:
- You may need to adjust the image tag and configuration paths depending on the CRS image version.
- Enable full audit logging in `modsecurity.conf` for rule ids and anomaly scores.
