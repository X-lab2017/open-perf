import json

with open('./meta-sample-data.json', 'r', encoding='utf-8') as file:
	data = json.load(file)

output_lines = []

for item in data:
	sentences = item.get("body", "")
	title = item.get("answer", "")

	if sentences and title:
		output_item = {
			"sentences": sentences,
			"title": title
		}
		output_lines.append(json.dumps(output_item, ensure_ascii=False))

with open('fine-turning-sample-data.json', 'w', encoding='utf-8') as file:
	for line in output_lines:
		file.write(line + '\n')

print(f"JSON data has been written")
