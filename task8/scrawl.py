import argparse
import requests
import pandas as pd
import time
import random
import json
import os
from tqdm import tqdm


def get_members(org_name, headers):
    members = []
    page = 1
    while True:
        print(f"Fetching members page: {page}")
        members_url = f"https://api.github.com/orgs/{org_name}/members?page={page}"
        response = requests.get(members_url, headers=headers)
        
        if response.status_code == 403:
            print("Rate limit exceeded, sleeping for 60 seconds")
            time.sleep(60)
            continue

        if response.status_code != 200:
            print(f"Error fetching members page {page}: {response.status_code}")
            break

        page_members = response.json()

        if not page_members:
            break

        members.extend(page_members)
        page += 1
        time.sleep(0.1)
        
    return members

def get_user_data_aux(username, headers):
    user_url = f"https://api.github.com/users/{username}"
    user_response = requests.get(user_url, headers=headers, timeout=10)
    user_response.raise_for_status()
    user_data = user_response.json()

    commits_url = f"https://api.github.com/search/commits?q=author:{username}+org:{org_name}"
    commits_response = requests.get(commits_url, headers=headers, timeout=10)
    commits_response.raise_for_status()
    commits_data = commits_response.json()
    commits_count = commits_data.get('total_count', 0)

    prs_url = f"https://api.github.com/search/issues?q=author:{username}+type:pr+org:{org_name}"
    prs_response = requests.get(prs_url, headers=headers, timeout=10)
    prs_response.raise_for_status()
    prs_data = prs_response.json()
    prs_count = prs_data.get('total_count', 0)

    issues_url = f"https://api.github.com/search/issues?q=author:{username}+type:issue+org:{org_name}"
    issues_response = requests.get(issues_url, headers=headers, timeout=10)
    issues_response.raise_for_status()
    issues_data = issues_response.json()
    issues_count = issues_data.get('total_count', 0)

    comments_url = f"https://api.github.com/search/issues?q=commenter:{username}+org:{org_name}"
    comments_response = requests.get(comments_url, headers=headers, timeout=10)
    comments_response.raise_for_status()
    comments_data = comments_response.json()
    comments_count = comments_data.get('total_count', 0)

    repos_url = f"https://api.github.com/search/repositories?q=org:{org_name}+fork:true+user:{username}"
    repos_response = requests.get(repos_url, headers=headers, timeout=10)
    repos_response.raise_for_status()
    repos_data = repos_response.json()
    repos_count = repos_data.get('total_count', 0)

    return {
        "Username": username,
        "Name": user_data.get('name'),
        "Company": user_data.get('company'),
        "Location": user_data.get('location'),
        "Public Repos": user_data.get('public_repos'),
        "Followers": user_data.get('followers'),
        "Following": user_data.get('following'),
        "Created At": user_data.get('created_at'),
        "Updated At": user_data.get('updated_at'),
        "Commits": commits_count,
        "Pull Requests": prs_count,
        "Issues": issues_count,
        "Comments": comments_count,
        "Repositories Contributed To": repos_count
    }

def get_user_data(username, headers):
    max_try = 5
    cnt = 0
    while True:
        try:
            if cnt == max_try:
                print("Max Try")
                return None
            return get_user_data_aux(username, headers)
        except requests.exceptions.RequestException as e:
            if e.response is None:
                print(f"Error fetching data for {username}: e is None")
                return None
            if e.response.status_code == 422:
                print(f"{e.response.status_code}: {e.response.json()}")
                return None
            wait_seconds = 10
            print(f"Error fetching data for {username}: {e}")
            print(f"Sleeping for {wait_seconds} seconds before retrying...")
            time.sleep(wait_seconds)
        cnt += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GitHub data scrawler")
    parser.add_argument("--org", type=str, help="Organization name", required=True)
    parser.add_argument("--token", type=str, help="GitHub API token", required=True)
    args = parser.parse_args()

    org_name = args.org

    token = args.token
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.cloak-preview"
    }

    # 存储文件路径
    data_file = f"{org_name}_users_detailed.csv"
    members_file = f"{org_name}_users.json"

    # 加载已经获取的成员数据
    if os.path.exists(members_file):
        with open(members_file, 'r') as f:
            members = json.load(f)
    else:
        members = []

    if not members:
        members = get_members(org_name, headers)
        with open(members_file, 'w') as f:
            json.dump(members, f)

    # 如果 CSV 文件不存在，创建包含标题行的文件
    if not os.path.exists(data_file):
        df = pd.DataFrame(columns=[
            "Username", "Name", "Company", "Location", "Public Repos",
            "Followers", "Following", "Created At", "Updated At",
            "Commits", "Pull Requests", "Issues", "Comments",
            "Repositories Contributed To"
        ])
        df.to_csv(data_file, index=False)

    # 从 CSV 文件中加载已爬取的用户数据
    if os.path.exists(data_file):
        df_existing = pd.read_csv(data_file)
        fetched_users = set(df_existing["Username"])
    else:
        fetched_users = set()

    # 单线程处理用户数据
    for member in tqdm(members, desc="Fetching user data"):
        username = member['login']
        if username not in fetched_users:
            print(f"Fetching data for {username}")
            result = get_user_data(username, headers)
            if result:
                fetched_users.add(username)
                df = pd.DataFrame([result])
                df.to_csv(data_file, mode='a', header=False, index=False)

            # 添加随机延迟以减少速率限制的发生
            time.sleep(random.uniform(1, 2))

    print("Data fetching complete.")

