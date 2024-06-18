import os
import requests
from datetime import datetime
from key import token

USERNAME = 'LeonardoANoman'
TOKEN = token  

API_URL = f'https://api.github.com/users/{USERNAME}/repos'

def fetch_repositories():
    headers = {
        'Authorization': f'token {TOKEN}'
    }
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch repositories: {response.status_code} - {response.text}")
        return None

def organize_repositories():
    repositories = fetch_repositories()
    if not repositories:
        return
    
    for repo in repositories:
        repo_name = repo['name']
        created_at = repo['created_at']
        year = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ').year
        repo_folder = f"{year}/{repo_name}"
        
        if not os.path.exists(str(year)):
            os.makedirs(str(year))
        
        os.system(f"git clone https://github.com/{USERNAME}/{repo_name}.git {repo_folder}")
        print(f"Cloned {repo_name} into {repo_folder}")


organize_repositories()
