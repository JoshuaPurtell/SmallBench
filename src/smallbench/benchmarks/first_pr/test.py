import base64
import requests


def get_repo_contents(repo_url, token=None):
    owner, repo = repo_url.split("/")[-2:]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()

    files = {}
    for item in response.json()["tree"]:
        if item["type"] == "blob" and item["path"].endswith(".py"):
            file_url = (
                f'https://api.github.com/repos/{owner}/{repo}/contents/{item["path"]}'
            )
            file_response = requests.get(file_url, headers=headers)
            file_response.raise_for_status()
            content = base64.b64decode(file_response.json()["content"]).decode("utf-8")
            files[item["path"]] = content

    return files


repo_url = "IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework"
token = "your_github_token"  # Optional
repo_files = get_repo_contents(repo_url, token)
