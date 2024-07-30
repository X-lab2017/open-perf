import requests
import pandas as pd
from datetime import datetime

# 设置仓库的基本信息
owner = "zhicheng-ning"
repo = "od-api"

# 获取提交信息
def get_commits(owner, repo):
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    response = requests.get(commits_url)
    commits = response.json()
    return commits

# 获取Pull Request信息
def get_pull_requests(owner, repo):
    prs_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all"
    response = requests.get(prs_url)
    prs = response.json()
    return prs

# 获取Issue信息
def get_issues(owner, repo):
    issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all"
    response = requests.get(issues_url)
    issues = response.json()
    return issues

# 处理提交信息
def process_commits(commits):
    commit_data = []
    for commit in commits:
        author = commit['commit']['author']['name']
        date = commit['commit']['author']['date']
        message = commit['commit']['message']
        commit_data.append({
            "author": author,
            "date": date,
            "message": message,
            "type": "code"
        })
    return commit_data

# 处理Pull Request信息
def process_pull_requests(prs):
    pr_data = []
    for pr in prs:
        author = pr['user']['login']
        created_at = pr['created_at']
        merged_at = pr['merged_at']
        pr_data.append({
            "author": author,
            "created_at": created_at,
            "merged_at": merged_at,
            "type": "review"
        })
    return pr_data

# 处理Issue信息
def process_issues(issues):
    issue_data = []
    for issue in issues:
        author = issue['user']['login']
        created_at = issue['created_at']
        closed_at = issue.get('closed_at', None)
        issue_data.append({
            "author": author,
            "created_at": created_at,
            "closed_at": closed_at,
            "type": "discussion"
        })
    return issue_data

# 获取数据
commits = get_commits(owner, repo)
prs = get_pull_requests(owner, repo)
issues = get_issues(owner, repo)

# 处理数据
commit_data = process_commits(commits)
pr_data = process_pull_requests(prs)
issue_data = process_issues(issues)

# 合并数据
collaboration_data = commit_data + pr_data + issue_data

# 打印协作数据
for data in collaboration_data:
    print(data)

## 写入csv
csv_file = 'collaboration_data.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=["author", "date", "message", "type", "created_at", "merged_at", "closed_at"])
    writer.writeheader()
    for data in collaboration_data:
        writer.writerow(data)

# 打印统计信息
print(f"\nTotal Commits: {len(commit_data)}")
print(f"Total Pull Requests: {len(pr_data)}")
print(f"Total Issues: {len(issue_data)}")


######### 第二种方式

# 设置仓库的基本信息
owner = "leveldb Team"
repo = "leveldb"

# 设置起始和结束日期
start_date = datetime(2019, 1, 1).isoformat()
end_date = datetime(2024, 7, 7).isoformat()

# 获取提交信息
def get_commits(owner, repo, start_date, end_date):
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {
        'since': start_date,
        'until': end_date
    }
    response = requests.get(commits_url, params=params)
    commits = response.json()
    return commits

# 获取Pull Request信息
def get_pull_requests(owner, repo, start_date, end_date):
    prs_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all"
    params = {
        'since': start_date,
        'until': end_date
    }
    response = requests.get(prs_url, params=params)
    prs = response.json()
    return prs

# 获取Issue信息
def get_issues(owner, repo, start_date, end_date):
    issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all"
    params = {
        'since': start_date,
        'until': end_date
    }
    response = requests.get(issues_url, params=params)
    issues = response.json()
    return issues

# 处理提交信息
def process_commits(commits):
    commit_data = []
    for commit in commits:
        author = commit['commit']['author']['name']
        date = commit['commit']['author']['date']
        message = commit['commit']['message']
        commit_data.append({
            "author": author,
            "date": date,
            "message": message,
            "type": "code"
        })
    return commit_data

# 处理Pull Request信息
def process_pull_requests(prs):
    pr_data = []
    for pr in prs:
        author = pr['user']['login']
        created_at = pr['created_at']
        merged_at = pr['merged_at']
        pr_data.append({
            "author": author,
            "created_at": created_at,
            "merged_at": merged_at,
            "type": "review"
        })
    return pr_data

# 处理Issue信息
def process_issues(issues):
    issue_data = []
    for issue in issues:
        author = issue['user']['login']
        created_at = issue['created_at']
        closed_at = issue.get('closed_at', None)
        issue_data.append({
            "author": author,
            "created_at": created_at,
            "closed_at": closed_at,
            "type": "discussion"
        })
    return issue_data

# all_commits = []
# all_prs = []
# all_issues = []
# # 获取数据（定期地）
# while true:
#     commits = get_commits(owner, repo, start_date, end_date)
#     all_commits.extend(commits)
    
#     prs = get_pull_requests(owner, repo, start_date, end_date)
#     all_prs.extend(prs)
    
#     issues = get_issues(owner, repo, start_date, end_date)
#     all_issues.extend(issues)

# 处理数据
commit_data = process_commits(commits)
pr_data = process_pull_requests(prs)
issue_data = process_issues(issues)

# 合并数据
collaboration_data = commit_data + pr_data + issue_data

# 打印协作数据
for data in collaboration_data:
    print(data)

## 写入csv
csv_file = 'collaboration_data2.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=["author", "date", "message", "type", "created_at", "merged_at", "closed_at"])
    writer.writeheader()
    for data in collaboration_data:
        writer.writerow(data)

# 打印统计信息
print(f"\nTotal Commits: {len(commit_data)}")
print(f"Total Pull Requests: {len(pr_data)}")
print(f"Total Issues: {len(issue_data)}")