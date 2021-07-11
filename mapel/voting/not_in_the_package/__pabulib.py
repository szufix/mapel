import csv

# IMPORT
def import_votes(path, vote_type="approval"):
    votes = {}
    file_ = open(path, 'r')
    lines = file_.readlines()

    if vote_type in ["approval", "ordinal"]:
        for line in lines:
            line = line.strip().split(',')
            votes[str(line[0])] = line[1:]
        return votes

    elif vote_type in ["cumulative", "scoring"]:
        for line in lines:
            line = line.strip().split(',')
            # moze slownik slowników
       # TBU


def import_voters(path):
    voters = {}
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            voters[row['pid']] = row['cost']
    return voters



def import_projects(path):
    costs = {}
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            costs[row['pid']] = row['cost']
    return costs


def import_info(path):
    info = {}
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            info[row['key']] = row['value']
    return info


# CONVERT OLD DATA
def convert_warsaw_data_to_new_format(district_name):

    # import old files

    file_name = "pabulib/projects/warsaw2020/" + district_name + ".txt"
    file_read = open(file_name, 'r')
    file_name = "pabulib/projects/warsaw2020/" + district_name + ".csv"
    file_write = open(file_name, 'w')

    num_lines = int(file_read.readline()) # skip
    file_write.write("pid,cost" + '\n')

    for _ in range(num_lines):
        line = file_read.readline()
        file_write.write(line)

    file_read.close()
    file_write.close()



    # save to new file