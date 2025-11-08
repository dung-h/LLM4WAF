import httpx, re
base='http://localhost:18081'
c=httpx.Client(follow_redirects=True)
r=c.get(base+'/setup.php')
print('GET setup backend:', r.status_code, 'len:', len(r.text))
m=re.search(r"name='user_token'\s+value='([^']+)'", r.text) or re.search(r'name=\"user_token\"\s+value=\"([^\"]+)\"', r.text)
token=m.group(1) if m else ''
print('token len:', len(token))
d={'create_db':'Create / Reset Database'}
if token:
    d['user_token']=token
resp=c.post(base+'/setup.php', data=d)
print('POST setup backend:', resp.status_code)
r2=c.get(base+'/setup.php')
print('Create present after?', 'Create / Reset Database' in r2.text, 'created msg?', 'Database has been created' in r2.text)
