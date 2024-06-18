import requests
from key import token

username = 'LeonardoANoman'

def delete_non_numeric_repos(username):
    # Fetch user repositories
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url, headers={"Authorization": f"token {token}"})
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            repo_name = repo['name']
            # Check if the repository name is non-numeric
            if not repo_name.isdigit():
                delete_repo(username, repo_name)
    else:
        print(f"Failed to fetch repositories for {username}, Status code: {response.status_code}")

def delete_repo(username, repository):
    # API URL to delete the repository
    api_url = f'https://api.github.com/repos/{username}/{repository}'
    response = requests.delete(api_url, headers={"Authorization": f"token {token}"})
    if response.status_code == 204:
        print(f"Deleted repository: {repository}")
    else:
        print(f"Failed to delete repository: {repository}, Status code: {response.status_code}")

# Start the process by fetching user repositories and deleting non-numeric ones
delete_non_numeric_repos(username)
