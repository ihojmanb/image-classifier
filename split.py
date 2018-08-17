import os
import sys
import errno
import csv
def count_csv_rows(csvfile):
    total_rows = 0
    with open(csvfile, 'rt') as file:
        row_count = sum(1 for row in csv.reader(file))
        total_rows = row_count
    file.close()    
    return total_rows

def split(csvfile,n_outputs ,outputname, output_dir, delimiter=',', keep_headers=False):

    output_name_template=outputname+'_%s.csv'
    current_dir = os.getcwd()
    print('current directory: ',current_dir)
    
    output_path=os.path.join(current_dir, output_dir)
    print('output_path: ',output_path)
    os.makedirs(output_path)
    n_rows = count_csv_rows(csvfile)
    row_limit = n_rows/n_outputs
    

    filehandler = open(csvfile, 'r')
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(output_path,output_name_template  % current_piece)
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = next(reader)
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(output_path,output_name_template  % current_piece)
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)

print('copia la direccion del directorio donde esta guardado el dataset deepsat-sat6')
print('deberia ser algo de este estilo : /Users/mac/code/custom-cnn/deepsat-sat6')
path = input('pegalo aqui aqui:')

file_path_1 = path +'/X_train_sat6.csv'
file_path_2 = path +'/y_train_sat6.csv'
file_path_3 = path +'/X_test_sat6.csv'
file_path_4 = path +'/y_test_sat6.csv'


print('espera un momento..')
print(file_path_1)
print(file_path_2)
print(file_path_3)
print(file_path_4)

split(file_path_1, n_outputs=10, outputname='X_train', output_dir='train_images')
split(file_path_2, n_outputs=10, outputname='train_labels', output_dir='train_labels')
split(file_path_3, n_outputs=10, outputname='X_test', output_dir='test_images')
split(file_path_4, n_outputs=10, outputname='test_labels', output_dir='test_labels')
print('listo!')
