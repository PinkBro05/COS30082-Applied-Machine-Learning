import re

def remove_version_constraints(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        # Remove version constraints
        updated_line = re.sub(r'==\S+|@\s*file://\S+', '', line).strip()
        if updated_line:
            updated_lines.append(updated_line + '\n')

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

if __name__ == "__main__":
    file_path = 'c:\\Users\\pink\\Documents\\Study\\Self-Study\\NLP_Course\\Blog-title-classification-using-transformer\\requirements.txt'
    remove_version_constraints(file_path)