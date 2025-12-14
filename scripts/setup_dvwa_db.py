import httpx
import re

client = httpx.Client(follow_redirects=True)
resp = client.get("http://localhost:8000/dvwa/setup.php")
token_match = re.search(r'user_token.*?value=["\']([a-f0-9]+)', resp.text)
if token_match:
    token = token_match.group(1)
    print(f"Token: {token}")
    result = client.post(
        "http://localhost:8000/dvwa/setup.php",
        data={"create_db": "Create / Reset Database", "user_token": token},
        follow_redirects=False  # Avoid redirect to /setup.php (outside /dvwa/)
    )
    print(f"POST status: {result.status_code} | Location: {result.headers.get('location')}")
    if result.status_code in (302, 301) and result.headers.get("location", "").rstrip("/").endswith("setup.php"):
        print("✅ Database create request accepted (DVWA returns 302).")
    elif "created" in result.text.lower() or "successfully" in result.text.lower():
        print("✅ Database created successfully!")
    else:
        print("❓ Database setup may not have completed, please check login/setup page.")
else:
    print("❌ Failed to extract user_token from setup page")
