import csv
import json

csv_file = './meta-sample-data.csv'
json_file = 'fine-turning-sample-data.json'

json_data = []

def clean_sentences(sentences):
	sentences = sentences.strip('[]')
	sentences = sentences.replace("'", "").replace(",", "")
	return sentences

with open(csv_file, mode='r', encoding='utf-8') as file:
	reader = csv.DictReader(file)
	for row in reader:
		cleaned_sentences = clean_sentences(row['title'])
		json_entry = {
			"sentences": cleaned_sentences,
			"labels": {
				"default": [row['category']]
			}
		}
		json_data.append(json_entry)

with open(json_file, mode='w', encoding='utf-8') as file:
	for entry in json_data:
		json_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
		file.write(json_line + '\n')

print(f"JSON data has been written to {json_file}")
