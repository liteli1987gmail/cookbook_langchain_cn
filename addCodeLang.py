import os

def replace_code_blocks(directory):
    in_code_block = False  # Track if we're inside a code block
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mdx"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding="utf-8") as f:
                    lines = f.readlines()

                with open(filepath, 'w', encoding="utf-8") as f:
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line == "```":  # This line is a code block delimiter
                            if in_code_block:  # If it's the end of a code block
                                in_code_block = False
                                f.write(line)
                            else:  # If it's the start of a code block
                                in_code_block = True
                                f.write("```python\n")
                        else:
                            f.write(line)

# 使用函数，例如遍历 'my_directory' 文件夹
replace_code_blocks('docs')
