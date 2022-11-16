import csv, pandas


def make_table(name, data, Q):
    header = ['q' + str(x) for x in range(Q)]
    with open(name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

def get_table_query(csv_name, col):
    col = "q" + str(col)
    col_list = [col]
    df = pandas.read_csv(csv_name, usecols=col_list)
    return list(df[col])


# make_table()
# ll = get_table_column('countries.csv', 0)
# q_rank = []

# q_rank.extend(ll)

# print(q_rank)
