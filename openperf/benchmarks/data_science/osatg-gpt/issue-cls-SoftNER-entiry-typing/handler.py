import json

def process_text_file(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	blocks = []
	current_block = []

	for line in lines:
		line = line.strip()
		if line == '':
			if current_block:
				blocks.append(current_block)
				current_block = []
		else:
			current_block.append(line)
	if current_block:
		blocks.append(current_block)

	results = []

	for block in blocks:
		sentences = []
		spans = []
		offset = 0

		for line in block:
			parts = line.split('\t')
			word = parts[0]
			tag = parts[2] if len(parts) > 2 else 'O'

			sentences.append(word)
			if tag != 'O':
				tag_type = tag.split('-')[-1].replace('_', ' ')
				start = offset
				end = start + len(word)
				spans.append({
					"start": start,
					"end": end,
					"type": tag_type,
					"term": word
				})

			offset += len(word) + 1

		if spans:
			sentence_str = ' '.join(sentences)
			results.append({
				"sentences": sentence_str,
				"spans": spans
			})

	with open('fine-turning-sample-data.json', 'w', encoding='utf-8') as f:
		for result in results:
			json.dump(result, f, ensure_ascii=False)
			f.write('\n')

process_text_file('./meta-sample-data.txt')
print(f"JSON data has been written")
