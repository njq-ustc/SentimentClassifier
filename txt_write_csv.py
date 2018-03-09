import csv

def get_lines(filepath):
    with open(filepath) as file_object:
        lines = set(file_object.readlines())
        return lines

def new_csv(lines):
    for line in lines:
        oneline = line.strip('\n')  # 逐行读取，剔除空白
        data = []
        data.append([oneline])
        with open(r'..\test\sentiment-analysis\positive.csv', 'a+') as csvfile:
            csv_writer = csv.writer(csvfile, dialect='excel')
            csv_writer.writerows(data)

if __name__ == "__main__":
    filepath = r"..\test\sentiment-analysis\positive.txt"
    lines = get_lines(filepath)
    new_csv(lines)