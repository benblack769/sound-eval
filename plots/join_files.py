import os

def get_line_num(line):
    return int(line.split()[0])

def join_similar_data(fnamelist,outname):
    if len(fnamelist) == 0:
        return
    with open(fnamelist[0]) as startfile:
        outlines = [line.strip() for line in startfile.readlines()]

    cur_end = get_line_num(outlines[-1])
    for filename in fnamelist[1:]:
        with open(filename) as nextfile:
            nextlines = nextfile.readlines()
        for i in range(len(nextlines)):
            linesp = nextlines[i].split()
            next_num = int(linesp[0]) + cur_end
            nextlines[i] = str(next_num) + "\t" + "\t".join(linesp[1:])

        outlines += nextlines

        cur_end = get_line_num(outlines[-1])
    filestr = "\n".join(outlines)
    with open(outname,"w") as outfile:
        outfile.write(filestr)
def join_folders_data(folder_list):
    out_files = dict()
    for fold in folder_list:
        for filename in os.listdir(fold):
            if filename in out_files:
                out_files[filename].append(fold)
            else:
                out_files[filename] = [fold]

    parent_dir = "plot_data/joined_data/"
    os.makedirs(parent_dir, exist_ok=True)
    for fname,fold_list in out_files.items():
        fnamelist = [os.path.join(fold,fname) for fold in fold_list]
        join_similar_data(fnamelist,parent_dir+fname)

def join_all():
    parent_dir = "plot_data/"
    fold_name = "layer501layer402tanh_layer34_train_test"
    fold_names = [fold_name+"0"*i for i in range(0,3)]
    #fold_names = ["joined_data","import_data2"]
    fold_paths = [parent_dir+name for name in fold_names]
    join_folders_data(fold_paths)
join_all()
