import csv
import json

csv_file = './meta-sample-data.csv'
json_file = 'fine-turning-sample-data.json'

json_data = []

with open(csv_file, mode='r', encoding='utf-8') as file:
	reader = csv.DictReader(file)
	for row in reader:
		json_entry = {
		"sentences": row['title'],
			"labels": {
				"default": [row['labels']]
			}
		}

		json_data.append(json_entry)

with open(json_file, mode='w', encoding='utf-8') as file:
	for entry in json_data:
		json_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
		file.write(json_line + '\n')

print(f"JSON data has been written {json_file}")
