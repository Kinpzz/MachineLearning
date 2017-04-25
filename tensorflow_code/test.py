import csv
# write data
with open( './data.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['id','reference'])
    for i in range(25):
        data = [i] + [i]
        writer.writerow(data)