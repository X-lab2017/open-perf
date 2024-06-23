import csv
import json

csv_file = './meta-sample-data.csv'
json_file = 'fine-turning-sample-data.json'

json_data = []

def convert_label(instance_type):
	if instance_type == '0':
		return 'negative'
	elif instance_type == '1':
		return 'positive'
	else:
		return instance_type

with open(csv_file, mode='r', encoding='utf-8') as file:
	reader = csv.DictReader(file)
	for row in reader:
		converted_label = convert_label(row['instance_type'])
		json_entry = {
			"sentences": row['comment_sentence'],
			"labels": {
				"default": [converted_label]
			}
		}

		json_data.append(json_entry)

with open(json_file, mode='w', encoding='utf-8') as file:
	for entry in json_data:
		json_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
		file.write(json_line + '\n')

print(f"JSON data has been written to {json_file}")
