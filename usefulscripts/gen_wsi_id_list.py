import csv


def get_wsi_id_labels(csv_file_path, survived_days=1825):
    with open(csv_file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        header = None
        wsi_labels = dict()
        for row in spamreader:
            if header is None:
                header = row[0]
            else:
                wsi_id = row[6][1:-1]
                if row[3] == 'NA':
                    continue
                days = int(row[3])
                dead = int(row[4][1:-1])
                if days <= survived_days and dead == 1:
                    wsi_labels[wsi_id] = 1
                if days > survived_days:
                    wsi_labels[wsi_id] = 0

    return wsi_labels


label_file_path = '/home/huidong/projects/BMI_projects/micnn_survival_rate/datasets/dataset_for_survival.csv'
# wsi_root = '/data03/shared/huidong/BMI_project/brca_data/WSIs_patches'
res = get_wsi_id_labels(label_file_path)

fn = './wsi_list.txt'
with open(fn, 'a') as f:
    for wsi_id in res.keys():
        wsi_path = '{}\n'.format(wsi_id)
        f.write(wsi_path)
        
    

