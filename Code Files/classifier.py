import os, re
from decimal import Decimal
from string import punctuation, digits
import math
import random
import argparse

###
# Method in order to tokenize line.
# It replaces some character with space.
###
def tokenize(text):
  text = text.lower()
  text = text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))
  text = text.translate(str.maketrans('', '', digits))
  text = re.sub(r'\s+', ' ', text).strip()

  return text
  
###
# Methods in order to read each train review, and fill multinomial dictionary.
###
def extractWordsMultinomial(file_name, negative):
  multinomial = negative_multinomial if negative else positive_multinomial
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = tokenize(line)
      words = re.findall(r'\w+', line)
      for word in words:
        multinomial[word] = multinomial.get(word, 0) + 1

###
# Methods in order to read each train review, and fill bernoulli dictionary.
###
def extractWordsBernoulli(file_name, negative):
  bernoulli = negative_bernoulli if negative else positive_bernoulli
  unique_words = set()
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = tokenize(line)
      words = re.findall(r'\w+', line)
      for word in words:
        unique_words.add(word)

  for word in unique_words:
    bernoulli[word] = bernoulli.get(word, 0) + 1

###
# Methods in order to read each train review, and fill binary dictionary.
###
def extractWordsBinary(file_name, negative):
  binary = negative_binary if negative else positive_binary
  unique_words = set()
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = tokenize(line)
      words = re.findall(r'\w+', line)
      for word in words:
        unique_words.add(word)

  for word in unique_words:
    binary[word] = binary.get(word, 0) + 1


###
# Methods in order to train reviews, it calls necessary algorithms.
###
def handleFiles(path):
  for folder in os.listdir(path):
    if folder == 'pos':
      number_of_positive_files = len(os.listdir(os.path.join(path, folder)))
    
    if folder == 'neg':
      number_of_negative_files = len(os.listdir(os.path.join(path, folder)))
    
    for file in os.listdir(os.path.join(path, folder)):
      if file.endswith('.txt'):
        extractWordsMultinomial(os.path.join(os.path.join(path, folder), file), folder == 'neg')
        extractWordsBernoulli(os.path.join(os.path.join(path, folder), file), folder == 'neg')
        extractWordsBinary(os.path.join(os.path.join(path, folder), file), folder == 'neg')

  return number_of_positive_files, number_of_negative_files

###
# Methods in order to test reviews in multinomial model.
###
def test_multinomial(file_name):
  pos = neg = 0
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = tokenize(line)
      words = re.findall(r'\w+', line)
      for word in words:
        pos = pos + math.log((positive_multinomial.get(word, 0) + 1) / (number_of_words_in_positive + number_of_unique_words))
        neg = neg + math.log((negative_multinomial.get(word, 0) + 1) / (number_of_words_in_negative + number_of_unique_words))

  pos = pos + math.log(number_of_positive_files / (number_of_positive_files + number_of_negative_files))
  neg = neg + math.log(number_of_negative_files / (number_of_positive_files + number_of_negative_files))

  return pos > neg

###
# Methods in order to test reviews in bernoulli model.
###
def test_bernoulli(file_name):
  pos = neg = 1
  unique_words = set()
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = tokenize(line)
      words = re.findall(r'\w+', line)
      for word in words:
        unique_words.add(word)

  for word in vocabulary:
    if word in unique_words:
      pos = pos * Decimal((positive_bernoulli.get(word, 0) + 1) / (number_of_positive_files + 2))
      neg = neg * Decimal((negative_bernoulli.get(word, 0) + 1) / (number_of_negative_files + 2))

    else:
      pos = pos * (1 - Decimal((positive_bernoulli.get(word, 0) + 1) / (number_of_positive_files + 2)))
      neg = neg * (1 - Decimal((negative_bernoulli.get(word, 0) + 1) / (number_of_negative_files + 2)))

  pos = Decimal(pos * number_of_positive_files / (number_of_positive_files + number_of_negative_files))
  neg = Decimal(neg * number_of_negative_files / (number_of_positive_files + number_of_negative_files))

  return pos > neg

###
# Methods in order to test reviews in binary model.
###
def test_binary(file_name):
  pos = neg = 0
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = tokenize(line)
      words = re.findall(r'\w+', line)
      for word in words:
        pos = pos + math.log((positive_binary.get(word, 0) + 1) / (number_of_words_in_positive_binary + number_of_unique_words))
        neg = neg + math.log((negative_binary.get(word, 0) + 1) / (number_of_words_in_negative_binary + number_of_unique_words))

  pos = pos + math.log(number_of_positive_files / (number_of_positive_files + number_of_negative_files))
  neg = neg + math.log(number_of_negative_files / (number_of_positive_files + number_of_negative_files))

  return pos > neg

###
# Methods in order to construct 2x2 table consisting of true positives, true negatives, false positives, false negatives.
###
def fillTable(positive_table, negative_table, pred_label, true_label):
  if pred_label == 1 and true_label == 1:
    positive_table[0][0] = positive_table[0][0] + 1
    negative_table[1][1] = negative_table[1][1] + 1

  elif pred_label == 1 and true_label == 0:
    positive_table[0][1] = positive_table[0][1] + 1
    negative_table[1][0] = negative_table[1][0] + 1

  elif pred_label == 0 and true_label == 1:
    positive_table[1][0] = positive_table[1][0] + 1
    negative_table[0][1] = negative_table[0][1] + 1

  else:
    positive_table[1][1] = positive_table[1][1] + 1
    negative_table[0][0] = negative_table[0][0] + 1


###
# Methods in order to test reviews, and calls necessary functions.
###
def test(path):
  counter_multinomial = counter_bernoulli = counter_binary = 0
  for folder in os.listdir(path):
    if folder == 'pos':
      number_of_test_positive_files = len(os.listdir(os.path.join(path, folder)))
    
    if folder == 'neg':
      number_of_test_negative_files = len(os.listdir(os.path.join(path, folder)))

    for file in os.listdir(os.path.join(path, folder)):
      if file.endswith('.txt'):
        label_multinomial = test_multinomial(os.path.join(os.path.join(path, folder), file))
        label_bernoulli = test_bernoulli(os.path.join(os.path.join(path, folder), file))
        label_binary = test_binary(os.path.join(os.path.join(path, folder), file))

        fillTable(table_multinomial_positive, table_multinomial_negative, label_multinomial, folder == 'pos')
        fillTable(table_bernoulli_positive, table_bernoulli_negative, label_bernoulli, folder == 'pos')
        fillTable(table_binary_positive, table_binary_negative, label_binary, folder == 'pos')

        if folder == 'pos':
          positive_multinomial_label.append(label_multinomial)
          positive_bernoulli_label.append(label_bernoulli)
          positive_binary_label.append(label_binary)
        else:
          negative_multinomial_label.append(label_multinomial)
          negative_bernoulli_label.append(label_bernoulli)
          negative_binary_label.append(label_binary)          

###
# Methods in order to calculate precision.
###
def calculate_precision(matrix):
  return matrix[0][0] / (matrix[0][0] + matrix[0][1])

###
# Methods in order to calculate recall.
###
def calculate_recall(matrix):
  return matrix[0][0] / (matrix[0][0] + matrix[1][0])

###
# Methods in order to calculate fscore.
###
def calculate_f_score(precision, recall):
  return (2 * precision * recall) / (precision + recall)

###
# Methods in order to construct microaveraged-table
###
def construct_microaveraged_table(first_table, second_table):
  table_microaveraged = [[0, 0], [0, 0]]

  table_microaveraged[0][0] = first_table[0][0] + second_table[0][0]
  table_microaveraged[0][1] = first_table[0][1] + second_table[0][1] 
  table_microaveraged[1][0] = first_table[1][0] + second_table[1][0] 
  table_microaveraged[1][1] = first_table[1][1] + second_table[1][1]

  return table_microaveraged

###
# Methods in order to report performance values of precision, recall, fscore for each algorithm.
###
def statistics(first_class, second_class, title):
  print('Statistics for ' + title + ' Naive Bayes Classifier')

  precision_positive = float('{0:.3f}'.format(calculate_precision(first_class)))
  recall_positive = float('{0:.3f}'.format(calculate_recall(first_class)))
  f_score_positive = float('{0:.3f}'.format(calculate_f_score(precision_positive, recall_positive)))

  precision_negative = float('{0:.3f}'.format(calculate_precision(second_class)))
  recall_negative = float('{0:.3f}'.format(calculate_recall(second_class)))
  f_score_negative = float('{0:.3f}'.format(calculate_f_score(precision_negative, recall_negative)))

  table_microaveraged = construct_microaveraged_table(first_class, second_class)
  precision_microaveraged = float('{0:.3f}'.format(calculate_precision(table_microaveraged)))
  recall_microaveraged = float('{0:.3f}'.format(calculate_recall(table_microaveraged)))
  f_score_microaveraged = float('{0:.3f}'.format(calculate_f_score(precision_microaveraged, recall_microaveraged)))

  precision_macroaveraged = float('{0:.3f}'.format((precision_positive + precision_negative) / 2))
  recall_macroaveraged = float('{0:.3f}'.format((recall_positive + recall_negative) / 2))
  f_score_macroaveraged = float('{0:.3f}'.format((f_score_positive + f_score_negative) / 2))


  metrics = ['precision', 'recall', 'f-score']
  data = [[precision_positive, recall_positive, f_score_positive], [precision_negative, recall_negative, f_score_negative], [precision_microaveraged, recall_microaveraged, f_score_microaveraged], [precision_macroaveraged, recall_macroaveraged, f_score_macroaveraged]]
  row_format ="{:>15}" * (len(metrics) + 1)
  print(row_format.format("", *metrics))
  for metric, row in zip(['positive', 'negative', 'micro-averaged', 'macro-averaged'], data):
    print(row_format.format(metric, *row))

  print('\n')
  print('- - - - - -')
  print('\n')

  return f_score_microaveraged

###
# Methods in order to perform randomization test.
###
def randomization_test(f_score_first, f_score_second, first_pred_pos, first_pred_neg, second_pred_pos, second_pred_neg):
  counter = 0.0

  f_score_diff = abs(f_score_first - f_score_second)
  R = 1000
  for _ in range(R):
    first_temp_pos = [[0, 0], [0, 0]]
    first_temp_neg = [[0, 0], [0, 0]]
    second_temp_pos = [[0, 0], [0, 0]]
    second_temp_neg = [[0, 0], [0, 0]]

    for first_pred_p, second_pred_p, first_pred_n, second_pred_n in zip(first_pred_pos, second_pred_pos, first_pred_neg, second_pred_neg):
      if random.random() > 0.5:
        fillTable(first_temp_pos, first_temp_neg, second_pred_p, True)
        fillTable(second_temp_pos, second_temp_neg, first_pred_p, True)
        fillTable(first_temp_pos, first_temp_neg, second_pred_n, False)
        fillTable(second_temp_pos, second_temp_neg, first_pred_n, False)

      else:
        fillTable(first_temp_pos, first_temp_neg, first_pred_p, True)
        fillTable(second_temp_pos, second_temp_neg, second_pred_p, True)
        fillTable(first_temp_pos, first_temp_neg, first_pred_n, False)
        fillTable(second_temp_pos, second_temp_neg, second_pred_n, False)

    
    first_micro = construct_microaveraged_table(first_temp_pos, first_temp_neg)
    first_precision = calculate_precision(first_micro)
    first_recall = calculate_recall(first_micro)
    first_f_score = calculate_f_score(first_precision, first_recall)
    
    second_micro = construct_microaveraged_table(second_temp_pos, second_temp_neg)
    second_precision = calculate_precision(second_micro)
    second_recall = calculate_recall(second_micro)
    second_f_score = calculate_f_score(second_precision, second_recall)
    
    current_f_score_diff = abs(first_f_score - second_f_score)
    if current_f_score_diff >= f_score_diff:
      counter = counter + 1

  p_value = (counter + 1) / (R + 1)

  return p_value

def randomization_test_init(f_score_first, f_score_second, alg1, alg2, first_pred_pos, first_pred_neg, second_pred_pos, second_pred_neg):
  print('Performing randomization test on ' +  alg1 + ' and ' + alg2)

  p_value = randomization_test(f_score_first, f_score_second, first_pred_pos, first_pred_neg, second_pred_pos, second_pred_neg)
  print('P-value is  {0:.3f}'.format(p_value))
  print('Both systems are not different. Difference is occurred by chance.') if p_value > 0.05 else print('Both systems are different. Difference is real.')

  print('\n')
  print('- - - - - -')
  print('\n')

###
# Parses arguments.
###
def parseArg():
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset', action="store", dest="dataset", required = True)
  return parser.parse_args()
  
positive_multinomial = {}
negative_multinomial = {}
positive_bernoulli = {}
negative_bernoulli = {}
positive_binary = {}
negative_binary = {}
table_multinomial_positive = [[0, 0], [0, 0]]
table_multinomial_negative = [[0, 0], [0, 0]]
table_bernoulli_positive = [[0, 0], [0, 0]]
table_bernoulli_negative = [[0, 0], [0, 0]]
table_binary_positive = [[0, 0], [0, 0]]
table_binary_negative = [[0, 0], [0, 0]]
positive_multinomial_label = []
negative_multinomial_label = []
positive_bernoulli_label = []
negative_bernoulli_label = []
positive_binary_label = []
negative_binary_label = []


arguments = parseArg()

dataset = arguments.dataset

number_of_positive_files, number_of_negative_files = handleFiles(os.path.join(dataset, 'train'))

vocabulary = set(positive_multinomial.keys()).union(negative_multinomial.keys())

number_of_unique_words = len(vocabulary)
number_of_words_in_positive = sum(positive_multinomial.values())
number_of_words_in_negative = sum(negative_multinomial.values())

number_of_words_in_positive_binary = sum(positive_binary.values())
number_of_words_in_negative_binary = sum(negative_binary.values())

test(os.path.join(dataset, 'test'))

f_score_multinomial = statistics(table_multinomial_positive, table_multinomial_negative, 'Multinomial')
f_score_bernoulli = statistics(table_bernoulli_positive, table_bernoulli_negative, 'Bernoulli')
f_score_binary = statistics(table_binary_positive, table_binary_negative, 'Binary')

randomization_test_init(f_score_multinomial, f_score_bernoulli, 'Multinomial NB', 'Bernoulli NB', positive_multinomial_label, negative_multinomial_label, positive_bernoulli_label, negative_bernoulli_label)
randomization_test_init(f_score_multinomial, f_score_binary, 'Multinomial NB', 'Binary NB', positive_multinomial_label, negative_multinomial_label, positive_binary_label, negative_binary_label)
randomization_test_init(f_score_bernoulli, f_score_binary, 'Bernoulli NB', 'Binary NB', positive_bernoulli_label, negative_bernoulli_label, positive_binary_label, negative_binary_label)
