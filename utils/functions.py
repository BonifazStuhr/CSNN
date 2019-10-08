import os
import sys
import urllib
import gzip
import shutil

def downloadDataset(data_url, save_to, force=False):
    """
    Checks if the files are already downloaded, if not the dataset will bed downloaded.
    :param save_to: (String) The directory in which the downloaded will be saved.
    :param force: (Boolean) If true, the download will be forced. False by default.
    """
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    downloadAndUncompressZip(data_url, save_to, force)

def downloadAndUncompressZip(url, dataset_dir, force=False):
    """
    :param URL: (String) The download link for the data.
    :param dataset_dir: (String) The save path of the data.
    :param force: (Boolean) Re-download data?
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    extract_to = os.path.splitext(filepath)[0]

    def downloadProgress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) *
                                                         100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
    else:
        filepath, _ = urllib.request.urlretrieve(url, filepath, downloadProgress)
        print()
        print('Successfully Downloaded', filename)

    # with zipfile.ZipFile(filepath) as fd:
    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
        print('Extracting ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('Successfully extracted')
        print()





