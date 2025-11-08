import httpx, re
base='http://localhost:8080'
c=httpx.Client(follow_redirects=True)
r=c.get(base+'/login.php')
print('GET login:', r.status_code, len(r.text))
m=re.search(r'name=\"user_token\"\s+value=\"([^\"]+)\"', r.text)
token=m.group(1) if m else ''
print('token?', bool(token))
resp=c.post(base+'/login.php', data={'username':'admin','password':'password','Login':'Login', **({'user_token':token} if token else {})})
print('POST login:', resp.status_code, 'url:', resp.url)
print('Set-Cookie headers:', resp.headers.get('set-cookie','')[:200])
resp2=c.get(base+'/index.php')
print('GET index:', resp2.status_code, 'len:', len(resp2.text))
