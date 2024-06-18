input_file_path = '14/chat.txt'
output_file_path = '14/output.txt'

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = file.read()

lines = data.split('\n')

modified_lines = []
for line in lines:
    parts = line.split(':')
    if len(parts) > 2 and ',' in parts[0]:
        date = parts[0]
        time = parts[1]
        user = parts[2].split(" ")[1] 
        msg = parts[3]
        modified_lines.append(f"{date}:{time} - {user}: {msg}")
    else:
        modified_lines.append(line)

modified_data = '\n'.join(modified_lines)

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(modified_data)

print(f'Transformation complete. Modified content saved to {output_file_path}')
