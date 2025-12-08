import httpx
import re

client = httpx.Client(follow_redirects=True)
resp = client.get("http://localhost:8000/modsec_dvwa/setup.php")
token_match = re.search(r'user_token.*?value=["\']([a-f0-9]+)', resp.text)
if token_match:
    token = token_match.group(1)
    result = client.post(
        "http://localhost:8000/modsec_dvwa/setup.php",
        data={"create_db": "Create / Reset Database", "user_token": token}
    )
    if "created" in result.text.lower() or "successfully" in result.text.lower():
        print("✅ Database created successfully!")
    else:
        print(f"❓ Database setup returned {result.status_code}")
        print("Check if it worked by trying to login")
else:
    print("❌ Failed to extract user_token from setup page")
