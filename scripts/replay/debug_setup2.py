import httpx, re
base='http://localhost:8080'
c=httpx.Client(follow_redirects=True)
r=c.get(base+'/setup.php')
m=re.search(r"name='user_token'\s+value='([^']+)'", r.text) or re.search(r'name=\"user_token\"\s+value=\"([^\"]+)\"', r.text)
token=m.group(1) if m else ''
print('token len:', len(token))
resp=c.post(base+'/setup.php', data={'create_db':'Create / Reset Database','user_token':token} if token else {'create_db':'Create / Reset Database'})
print('POST setup:', resp.status_code)
r2=c.get(base+'/setup.php')
print('Create present after?', 'Create / Reset Database' in r2.text, 'created msg?', 'Database has been created' in r2.text)
