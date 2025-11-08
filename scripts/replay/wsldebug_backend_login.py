import httpx, re
base='http://localhost:18081'
c=httpx.Client(follow_redirects=True)
r=c.get(base+'/login.php')
print('GET login backend:', r.status_code, len(r.text))
m=re.search(r"name='user_token'\s+value='([^']+)'", r.text) or re.search(r'name=\"user_token\"\s+value=\"([^\"]+)\"', r.text)
token=m.group(1) if m else ''
print('token?', bool(token))
d={'username':'admin','password':'password','Login':'Login'}
if token:
    d['user_token']=token
resp=c.post(base+'/login.php', data=d)
print('POST backend:', resp.status_code, 'url:', resp.url)
print('set-cookie:', resp.headers.get('set-cookie','')[:150])
r2=c.get(base+'/index.php')
print('GET index backend:', r2.status_code, 'len:', len(r2.text), 'redir?', '/login.php' in str(r2.url))
