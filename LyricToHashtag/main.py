def convert_to_single_string(text):
    text = text.replace("'", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    words = text.split()
    single_string = ''.join(word.capitalize() for word in words)
    
    return single_string

text = """
Just a simple,
example of usage,
I'm sure it'll work.
"""

single_string = convert_to_single_string(text)
print(single_string)
