from glob import glob
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
seg_path = '/data02/shared/huidong/BMI_project/brca_data/Nuclei_segs/brca_prob'

# wsi_root = '/data03/shared/huidong/BMI_project/brca_data/WSIs_patches'
res = get_wsi_id_labels(label_file_path)

seg_path_list = glob('{}/*'.format(seg_path))


fn = './wsi_list_seg.txt'
with open(fn, 'a') as f:
    for idx, seg_path in enumerate(seg_path_list, 0):
        seg_fn = seg_path.split('/')[-1]
        seg_fn_id = seg_fn.split('.')[0]
        id_exists = seg_fn_id in res

        if id_exists:
          f.write('{}\n'.format(seg_fn))  
        
        
    

