import pickle
import random
from collections import namedtuple, Counter
from typing import Tuple
import matplotlib.pyplot as plot
import cv2
# import lmdb
import numpy as np
import pandas
from path import Path
import os
import string

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoader:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size = 100,
                 data_split: float = 0.95,
                #  data_split: float = 0.0, # para cogerlo todo para validar
                 fast: bool = False,
                 line_mode: bool = False,
                 task_iam: bool = False,
                 task_cenatav: bool = False,
                 simulate: bool = False) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        self.line_mode = line_mode
        # if fast:
        #     self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

############# para las palabras
        if simulate or not self.line_mode:
            f = open(data_dir / 'gt/words.txt')  # para para IAM
    #        f = open(data_dir / 'gt_CVL/words_CVL.txt') # para CVL, se lo a;adio Carballea
            chars = set()
            words = set()
            # bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
            for line in f:
                # ignore comment line
                if not line or line[0] == '#':
                    continue
    
                line_split = line.strip().split(' ')
                # assert len(line_split) >= 9 # para para IAM
                assert len(line_split) >= 6 # para para CENATAV # Carballea
    #            assert len(line_split) >= 6 # para CVL, se lo a;adio Carballea
    
                # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
                file_name_split = line_split[0].split('-')
                file_name_subdir1 = file_name_split[0]
                file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}-{file_name_split[2]}'
                file_base_name = line_split[0] + '.png'
                file_name = data_dir / 'words' / file_name_subdir1 / file_name_subdir2 / file_base_name # para para IAM
    #            file_name = data_dir / 'img_CVL' / file_name_subdir1 / file_name_subdir2 / file_base_name # para CVL, se lo a;adio Carballea
    
                # if line_split[0] in bad_samples_reference:
                #    print('Ignoring known broken image:', file_name)
                #    continue
    
                # GT text are columns starting at 5
                gtText = ' '.join(line_split[5:])
                chars = chars.union(set(list(gtText)))
                
                
                # filter punctuation signs and numbers of the words
                new_gtText_list = []
                char_gt_list = []
                for c in gtText:
                    if c not in string.punctuation + string.digits:
                        char_gt_list.append(c)
                
                new_gt = "".join(char_gt_list)
                if new_gt != "":
                    new_gtText_list.append(new_gt.lower())
                    words = words.union(set(list(new_gtText_list)))
    
                # put sample into list
                self.samples.append(Sample(gtText, file_name))

################ para las lineas
        else:
            f = open(data_dir / 'gt/lines.txt')
            chars = set()
            chars_array = []
            words = set()
            words_array = []
            bad_samples = []
#            bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
            bad_samples_reference = ['']
            for line in f:
                # ignore comment line
                if not line or line[0] == '#':
                    continue
    
                line_split = line.strip().split(' ')  ## remove the space and split with ' '
                try:
                    assert len(line_split) >= 6
                except:
                    print(f'Error in the line: {line_split}')
    
                # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
                file_name_split = line_split[0].split('-')
                #print(fileNameSplit)
                file_base_name = Path(line_split[0] + '.png')
                file_name_subdir1 = file_name_split[0]
                file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}-{file_name_split[2]}'
                fileName = data_dir / 'lines' / file_name_subdir1 / file_name_subdir2 / file_base_name
    
                # GT text are columns starting at 5
                # see the lines.txt and check where the GT text starts, in this case it is 5
                gtText_list = line_split[5].split('|')
                gtText = ' '.join(gtText_list)
                chars = chars.union(set(list(gtText)))  ## taking the unique characters present
                
                # filter punctuation signs and numbers of the words
                new_gtText_list = []
                for gt in gtText_list:
                    char_gt_list = []
                    for c in gt:
                        if c not in string.punctuation + string.digits:
                            char_gt_list.append(c)
                        
                        chars_array.append(c)
                            
                    new_gt = "".join(char_gt_list)
                    if new_gt == "":
                        continue
                    new_gtText_list.append(new_gt.lower())
                    words_array.append(new_gt)
                
                words = words.union(set(list(new_gtText_list)))
                
                # check if image is not empty
                if not os.path.getsize(fileName):
                    bad_samples.append(line_split[0] + '.png')
                    continue
    
                # put sample into list
                self.samples.append(Sample(gtText, fileName))
    
    
            # some images in the IAM dataset are known to be damaged, don't show warning for them
            if set(bad_samples) != set(bad_samples_reference):
                print("Warning, damaged images found:", bad_samples)
                print("Damaged images expected:", bad_samples_reference)

        if not task_cenatav:
            # split into training and validation set: 95% - 5%
            split_idx = int(data_split * len(self.samples))
            self.train_samples = self.samples[:split_idx]
            self.validation_samples = self.samples[split_idx:] # para para IAM

        else: # los de 6 formularios de entrenamiento y los de 3 de validacion
            # en summary aparece un listado por escritores y # de formularios completados
            
            self.list_validate = ['008', '009', '011', '012', '013',
                                  '017', '020', '021', '022', '023',
                                  '024']
            self.list_validate += ['366', '365', '364', '363', '362', '361', '360'] # escritores completos
            self.list_test = ['025', '026', '027', '029','030',
                              '031', '032', '033', '034', '036',
                              '037', '275', '276', '289', '290',
                              '301', '304', '326', '332', '348']
            self.list_test += ['000', '001', '002', '003', '004', '005', '006'] # escritores completos
            # los escritores que hayan llenado 6 formularios seran de entrenamiento
            # las linea presentes en los escritores de la lista hacen ~ 10% de validacion
            self.train_samples = []
            self.validation_samples = []
            self.test_samples = []
            for element in self.samples:
                if element.file_path.basename()[:3] not in (self.list_validate + self.list_test):
                # if element.file_path.basename()[:3] not in (list_validate):
                    self.train_samples.append(element)
                elif element.file_path.basename()[:3] in self.list_test:
                    self.test_samples.append(element)
                elif element.file_path.basename()[:3] in self.list_validate:
                    self.validation_samples.append(element)
                else:
                    print(f"*{element}* no pertenece a ningun conjunto")

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]
        self.test_words = [x.gt_text for x in self.test_samples]

        # print(f'***Total images: {len(self.train_samples)}***')
        print(f'Train images: {len(self.train_samples)}')
        print(f'Validate images: {len(self.validation_samples)}')
        print(f'Test images: {len(self.test_samples)}')
            
        # start with train set
        # self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))
        
        # list of all words in dataset
        self.word_list = sorted(list(words))

# In[] PLot Histogram
        # # plot histogram of characters
        # chars_counts = Counter(chars_array)
        # sortedDict = sorted(chars_counts.items(), key=lambda x: x[1], reverse=True)
        # chars_counts_sorted = {}
        # suma = 0
        # for i in range(50):
        # # for i in range(len(sortedDict)): # 
        #     key = sortedDict[i][0]
        #     if key == '.':
        #         key = '(dot) .'
        #     elif key == ',':
        #         key = '(comma) ,'
        #     elif key == '-':
        #         key = '(hyphen) -'
        #     elif key == '"':
        #         key = '(quote) "' 
        #     chars_counts_sorted[key] = sortedDict[i][1]
        #     suma += sortedDict[i][1]
        # prom = suma / len(sortedDict) # average number of occurrences of a character in the database
        # df_chars = pandas.DataFrame.from_dict(chars_counts_sorted, orient="index")
        
        # plot.rcParams.update({'font.size': 15})
        # df_chars.plot(kind='bar')
        # plot.title('Histogram of characters -- CENATAV-HTR Database')
        # plot.xlabel('Characters')
        # plot.ylabel('Frecuency')
        # plot.show()   
        
        # # plot histogram of words
        # words_counts = Counter(words_array)
        # sortedDict_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
        # words_counts_sorted = {}
        # for i in range(50):
        #     words_counts_sorted[sortedDict_words[i][0]] = sortedDict_words[i][1]
        # df_words = pandas.DataFrame.from_dict(words_counts_sorted, orient="index")
        
        # df_words.plot(kind='bar')
        # plot.title('Histogram of words -- CENATAV-HTR Database')
        # plot.xlabel('Words')
        # plot.ylabel('Frecuency')
        # plot.show()
# In[] End                

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        # random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))

def init_dataset(partitions):
    dataset = dict()
    for i in partitions:
        dataset[i] = {"dt": [], "gt": []}

    return dataset

def get_dataset(loader, partitions):
    dataset = init_dataset(partitions)
    dataset['train']['gt'] = [train_samples.gt_text for train_samples in loader.train_samples]
    dataset['train']['dt'] = [train_samples.file_path for train_samples in loader.train_samples]
    dataset['valid']['gt'] = [validation_samples.gt_text for validation_samples in loader.validation_samples]
    dataset['valid']['dt'] = [validation_samples.file_path for validation_samples in loader.validation_samples]
    dataset['test']['gt'] = [test_samples.gt_text for test_samples in loader.test_samples]
    dataset['test']['dt'] = [test_samples.file_path for test_samples in loader.test_samples]
    return dataset

def save_dataset(dataset, level, num_writers, scr_path, partitions):
    tasks_txt_path = os.path.join(scr_path, f"Writer_Independent_{level}_Recognition_Task")  
    os.makedirs(tasks_txt_path, exist_ok=True)   
    
    summary_txt_path = os.path.join(tasks_txt_path, f'Writer_Independent_{level}_Recognition_Task.txt')
    total = len(dataset["train"]["dt"])+len(dataset["test"]["dt"])+len(dataset["valid"]["dt"])
    summary = "\n".join([
            f'***Summary of Writer Independent {level} Recognition Task in CENATAV DataBase***',
            f'train_set.txt - {level}s: {len(dataset["train"]["dt"])} writers: {num_writers[2]-num_writers[0]-num_writers[1]}',
            f'test_set.txt  - {level}s: {len(dataset["test"]["dt"])} writers: {num_writers[0]}',
            f'valid_set.txt - {level}s: {len(dataset["valid"]["dt"])}  writers: {num_writers[1]}',
            f'Total:        - {level}s: {total} writers: {num_writers[2]}',
        ])
    with open(summary_txt_path, "w") as file:
        file.write(summary)
        print(summary)
        
    for p in partitions:
        txt_path = os.path.join(tasks_txt_path, f'{p}_set.txt')
        open(txt_path, "w").close()
        for dt in dataset[p]['dt']:
            img_name = dt.name.split(".")[0]
            with open(txt_path, "a") as file:
                file.write(img_name+'\n')
                    
if __name__ == '__main__':
    scr_path = os.path.join("F:\(0) trabajo CENATAV\\00_BDs", "CENATAV Database")
    save_txt = False
    total_writers = 152
    partitions = ['train', 'valid', 'test']
    line_mode = True
    
    loader = DataLoader(Path(scr_path), line_mode=line_mode,
                                            task_cenatav=True, simulate = False)
    # if self.simulate:
    #     loader_simulate = dataloader_cenatav.DataLoader(Path(self.source), line_mode=True,
    #                                         task_cenatav=True, simulate = True) 
    dataset = get_dataset(loader, partitions)
    
    if save_txt:
        level = 'Text_Line' if line_mode else "Word"
        save_dataset(dataset, level,
                     [len(loader.list_test), len(loader.list_validate), total_writers],
                     scr_path, partitions
                     )
    
    pass            
            