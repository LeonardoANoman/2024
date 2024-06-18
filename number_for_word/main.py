import random

file_path = "words.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    valid_words = set(file.read().split())

def generate_random_word(length):
    characters = ['a', 'e', 'i', 'o', 'u']
    diacritic_code_points = [
        [224, 225, 226, 227],  # 'à', 'á', 'â', 'ã'
        [232, 233, 234, 235],  # 'è', 'é', 'ê', 'ë'
        [236, 237, 238, 239],  # 'ì', 'í', 'î', 'ï'
        [242, 243, 244, 245],  # 'ò', 'ó', 'ô', 'õ'
        [249, 250, 251, 252]   # 'ù', 'ú', 'û', 'ü'
    ]

    include_diacritic = random.choice([True, False])
    diacritic_vowel = chr(random.choice(random.choice(diacritic_code_points))) if include_diacritic else ""
    remaining_length = length - len(diacritic_vowel)
    regular_vowel = random.choice(characters) if not include_diacritic else ""
    consonants = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in characters]
    random_consonants = [random.choice(consonants) for _ in range(remaining_length - 1)]
    additional_vowels = [random.choice(characters) for _ in range(remaining_length - len(random_consonants))]
    random_characters = [diacritic_vowel] + [regular_vowel] + random_consonants + additional_vowels
    random.shuffle(random_characters)
    random_word = ''.join(random_characters)
    return random_word

def is_valid(word):
    return word in valid_words

def len_was_generated(words_list):
    seen = set()
    lengths = [len(word) for word in words_list]
    for value in lengths:
        if value in seen:
            return True
        seen.add(value)
    return False

def generate_random_list(char_len, word_len):
    i = 0
    words_list = []
    generated_lengths = set()
    max_len = 0
    while len(words_list) < word_len:
        i += 1
        len_as_i = len(generated_lengths)
        word_length = random.randint(len_as_i + char_len, len_as_i + char_len + 1)
        random_word = generate_random_word(word_length)
        if is_valid(random_word) and not len_was_generated(words_list):
            words_list.append(random_word)
            generated_lengths.add(len(random_word))
            max_len = max_len + 1 if len(random_word) <= 2 else max_len
    return words_list, i

def first_join_valid(generated_lists):    
    for random_list in generated_lists:
        flattened_list = [item for sublist in random_list for item in sublist]
        joined_string = "".join(flattened_list)
        if is_valid(joined_string):
            return joined_string
    generated_lists.clear()

def play(x,y):
    generated_lists = []
    n = 0
    while True:
        random_list, i = generate_random_list(x,y)
        generated_lists.append(random_list)
        if first_join_valid(generated_lists):
            return first_join_valid(generated_lists), i
        n += 1

result, i = play(2, 2)
print(result)
print(i)