### 为了获得npm中提到的各个repo的贡献信息
import pandas as pd
import numpy as np
import csv
import requests
import re
import time
import json

file_path = "C:/Users/lenovo/Desktop/npm_refer_by_repo_2024-05-13-16-54-33_EXPORT_CSV_13529018_459_0.csv"
column_name = "repository_url" ## 读取repo的url

data = pd.read_csv(file_path)
repo_url = data[column_name]
repo_url = repo_url.dropna() ## 去掉nan
repo_url_list= repo_url.to_list()  ## 转成list
repo_url_list

## 提取出上述url的owner和repo名对（用以后续爬取贡献信息）
def get_owner_repo(url):
    # 定义正则表达式模式
    pattern = r'\/([^\/]+)\/([^\/]+)\.git(?:\/|$)'
    match = re.search(pattern, url)
    if match:
        o = match.group(1)
        r = match.group(2)
        return o, r       ## 可以返回出来（调用函数时，申明这对变量）
    else:
        return None, None

url = 'git+https://github.com/hfreire/request-on-steroids.git'
owner, repo = get_owner_repo(url)
print(owner, repo)

owner_repo_dict = {}   ## 形成owner和repo对应的字典
for url in repo_url_list:
    owner, repo = get_owner_repo(url)
    if owner != None:
        owner_repo_dict[owner] = repo
    print(owner, repo)
print(len(owner_repo_dict)) ## 打印看一下长度

### 接下来，对于list中的每个url，都提取贡献内容的数据
### 写入数据框
repo_developer_df = pd.DataFrame(columns=['repo_name', 'developers'])

# 设置请求头部，包括接受 JSON 格式和认证信息（使用个人访问令牌）
headers = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': 'token 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'
}
proxies = {
    'http': 'http://127.0.0.1:7890', 
    'https': 'http://127.0.0.1:7890'
}

# 设置最大尝试次数
max_attempts = 1
index = 0
count = 0 ## 用以访问计数

for owner, repo in owner_repo_dict.items():
    repo_url = f'https://api.github.com/repos/{owner}/{repo}/contributors'
    attempts = 0
    count += 1
    
    while attempts < max_attempts:
        try:
            response = requests.get(repo_url, headers=headers, proxies=proxies)
            
            #### 每1000次访问，持久化
            if count % 1000 == 0:
                repo_developer_df.to_csv("contributors_of_repo.csv", index=False, mode='w')
                print(f"爬虫已经访问了{count}次了..")
            #### 写入disk（全覆盖写）
            
            if response.status_code == 200:
            # 提取开发者名字
                developers = []
                for contributor in response.json():
                    developers.append(contributor['login'])
                print(developers)
                
                row = {'repo_name' : repo, 'developers' : developers}
                repo_developer_df.loc[index] = row.values()
                index += 1
                break
            else:
                print("该仓库status异常..") ## 要重试
            attempts += 1
            
        except requests.exceptions.RequestException as e:
            # 处理网络请求异常
            print(f"Error during request: {e}")
            attempts += 1
        except json.JSONDecodeError as e:
            # 处理 JSON 解析异常
            print(f"Error decoding JSON response: {e}")
            attempts += 1
        except Exception as e:
            # 处理其他未知异常
            print(f"An unexpected error occurred: {e}")
            attempts += 1
        
        
        if attempts < max_attempts:  # 如果未能成功请求，则等待一段时间再次尝试
            print(f"Retrying ({attempts}/{max_attempts})...")
            #time.sleep(0.1)  # 等待 1 秒再次尝试请求
        else:
            print(f"rertry已超过Max({max_attempts}). Exiting...")
            continue

## 写回csv（持久化于磁盘，方便使用）
repo_developer_df.to_csv("contributors_of_repo.csv", index=False, mode='w')
## 把repo数据再读回来
df = pd.read_csv("contributors_of_repo.csv")
print(df)