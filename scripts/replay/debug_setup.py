import httpx, re
base='http://localhost:8080'
c=httpx.Client(follow_redirects=True)
r=c.get(base+'/setup.php')
print('GET setup:', r.status_code, 'len:', len(r.text))
print('Has Create Reset?', 'Create / Reset Database' in r.text)
m=re.search(r'name=\"user_token\"\s+value=\"([^\"]+)\"', r.text)
token=m.group(1) if m else ''
print('token?', bool(token))
# Always attempt POST; include token if found
data={'create_db':'Create / Reset Database'}
if token:
    data['user_token']=token
r2=c.post(base+'/setup.php', data=data)
print('POST setup:', r2.status_code, 'len:', len(r2.text))
