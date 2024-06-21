import csv
import json

def process_csv_file(input_file_path, output_file_path):
	results = []

	with open(input_file_path, 'r', encoding='utf-8') as csvfile:
		csvreader = csv.reader(csvfile, delimiter='\t')

		for row in csvreader:
			if row:
				label_and_text = row[0].strip()
				if label_and_text.startswith('__label__'):
					label_start_idx = len('__label__')
					label_end_idx = label_and_text.find(' ', label_start_idx)
					if label_end_idx == -1:
						label_end_idx = len(label_and_text)
					label = label_and_text[label_start_idx:label_end_idx]
					sentence = label_and_text[label_end_idx:].strip()

					result = {
					"sentences": sentence,
						"labels": {
							"default": [label]
						}
					}
					results.append(result)

	with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
		for result in results:
			json.dump(result, jsonfile, ensure_ascii=False)
			jsonfile.write('\n')

process_csv_file('./meta-sample-data.csv', 'fine-turning-sample-data.json')
