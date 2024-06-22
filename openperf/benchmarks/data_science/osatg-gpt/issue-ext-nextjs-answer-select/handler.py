import json

# 读取原始的JSON文件
with open('./meta-sample-data.json', 'r', encoding='utf-8') as file:
	data = json.load(file)

output_lines = []

for item in data:
	sentences = item.get("body", "")
	comment_1 = item.get("comment_1", "")
	comment_2 = item.get("comment_2", "")
	comment_3 = item.get("comment_3", "")
	comment_4 = item.get("comment_4", "")
	comment_5 = item.get("comment_5", "")

	comments = [comment for comment in [comment_1, comment_2, comment_3, comment_4, comment_5] if comment]

	if len(comments) > 1 and sentences:
		output_item = {
			"sentences": sentences,
			"comments": comments
		}
		output_lines.append(json.dumps(output_item, ensure_ascii=False))

# 将生成的内容写入到新的JSON文件
with open('fine-turning-sample-data.json', 'w', encoding='utf-8') as file:
	for line in output_lines:
		file.write(line + '\n')

print("JSON data has been written")
