import os
import argparse

def make_items(item_list):
    return "</li>\n    <li>".join(item_list)

def make_link(filename):
    return '<a href="{}">{}</a>'.format(filename,filename)
def make_links(filenames):
    return [make_link(fname) for fname in filenames]

def save_string(filename,string):
    with open(filename,'w') as file:
        file.write(string)

def make_html(relpath,filenames):
    template_html = '''
    <!doctype html>
    <h2>Path: {relpath}</h2>
    <ul>
        <li>{items}</li>
    </ul>
    '''
    return template_html.format(relpath=relpath,items=make_items(make_links(filenames)))

def indexify_folder(base_folder, rel_folder="./"):
    assert base_folder[-1] == "/"
    
    path = base_folder + rel_folder
    sub_items = os.listdir(path)
    dirs_tail_slash = [item + "/" if os.path.isdir(path+item) else item for item in sub_items]
    save_string(path + "index.html",make_html(rel_folder,dirs_tail_slash))
    for item in sub_items:
        item_path = os.path.join(path,item)
        if os.path.isdir(item_path):
            indexify_folder(base_folder,rel_folder+item+"/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give a folder an index.html listing files recursively")
    parser.add_argument('folder', help='Path to folder')

    args = parser.parse_args()

    folder = args.folder
    if folder[-1] != "/":
        folder += "/"

    indexify_folder(folder)
