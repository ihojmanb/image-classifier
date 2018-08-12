import os
import csv
import errno

def count_csv_rows(csvfile):
    total_rows = 0
    with open(csvfile, 'rt') as file:
        row_count = sum(1 for row in csv.reader(file))
        total_rows = row_count
    file.close()    
    return total_rows

def split(csvfile,n_outputs ,outputname, output_dir, delimiter=',', keep_headers=False):
    """
    Splits a CSV file into multiple pieces.
    
    A quick bastardization of the Python CSV library.

    Arguments:
        'csvfile': path to your csvfile
        'n_outputs': number of output files that you want the file to be divided in. 
        `output_name: template name of output files
        `output_dir`: name of dir that will store the  output files.
        `keep_headers`: Whether or not to print the headers in each output file.

    Example usage:
    
        >> from toolbox import csv_splitter;
        >> csv_splitter.split(open('/home/ben/input.csv', 'r'));
    
    """
    output_name_template=outputname+'_%s.csv'
    current_dir = os.getcwd()
    print(current_dir)
    output_path=os.path.join(current_dir, output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_rows = count_csv_rows(csvfile)
    row_limit = n_rows/n_outputs

    filehandler = open(csvfile, 'r')
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(output_path,output_name_template  % current_piece)

    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(output_path, output_name_template  % (current_piece))
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)

split('/Users/mac/code/custom-cnn/deepsat-sat6/X_train_sat6.csv', n_outputs=10, outputname='X_train', output_dir='/new/train_images')
split('/Users/mac/code/custom-cnn/deepsat-sat6/y_train_sat6.csv', n_outputs=10, outputname='train_labels', output_dir='/new/train_labels')
split('/Users/mac/code/custom-cnn/deepsat-sat6/X_test_sat6.csv', n_outputs=10, outputname='X_test', output_dir='/new/test_images')
split('/Users/mac/code/custom-cnn/deepsat-sat6/y_test_sat6.csv', n_outputs=10, outputname='test_labels', output_dir='/new/test_labels')







