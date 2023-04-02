# -*- coding: utf-8 -*-
import codecs
import collections
import logging
import re
import pymorphy2
from nltk.tokenize import word_tokenize
import os
import sys
import math
import numpy
import pandas


logging.basicConfig(level=logging.DEBUG)
morph = pymorphy2.MorphAnalyzer()


os.makedirs('finished_topics')


def to_words(tokens):
    for w in tokens:
        w = re.sub('^[0-9]\.', '', w)
        w = re.sub('[a-zA-Z«»\.]', '', w)
        if len(w) == 1 and not w.isalpha() or w == '...' or w == '``' or w == '\'\'' or w.isnumeric() or len(
                w) == 0 or w == '':
            continue
        for sw in w.split('-'):
            yield sw

def hyphen_words(tokens):
    for w in tokens:
#         w = w.encode("utf-8").replace('«', '').decode("utf-8")
        for sw in w.split('-'):
            yield sw

def normalize(w):
    return (morph.parse(w))[0].normal_form


def find_bigrams(words, windowSize=2):
    for index, word in enumerate(words):
        for otherIndex in range(index - windowSize, index + windowSize + 1):
            if otherIndex >= 0 and otherIndex < len(words) and otherIndex != index:
                otherWord = words[otherIndex]
                p = tuple(sorted([word, otherWord]))
                yield p

def get_pmi_for_bigrams():
    with codecs.open(PMI_FILE, 'r', 'utf8') as bigrams_pmis_file:
        bigrams_pmis = dict()
        for line in bigrams_pmis_file.readlines():
            ls = line[:-1].split('\t')
            bigrams_pmis[(ls[0], ls[1])] = float(ls[2])
    return bigrams_pmis

def calculate_query2titles():
    with codecs.open(TITLES_FILE, 'r', 'utf8') as titles_file:
        q = titles_file.readline()[7:]
        titles = titles_file.readlines()[2:]

        yield (q, titles)


def calc_uniq_bigrams(query2titles):
    bigrams_uniq = set()
    for q, titles in query2titles:
        for title in titles:
            tokens = word_tokenize(title.lower())
            words = [w for w in to_words(tokens)]
            words = [w for w in map(normalize, words) if w not in STOP_WORDS]
            for bigram in find_bigrams(words):
                bigram = tuple(sorted(bigram))
                bigrams_uniq.add(bigram)
    return bigrams_uniq


def get_raw_titles(titles_src):
    raw_titles = []
    for title in titles_src:
        tokens = word_tokenize(title.lower())
        raw_words = [w for w in hyphen_words(tokens)]
        raw_titles.append(raw_words)
    return raw_titles


def calculate_graph(query, titles_src, bigrams_pmis, mode_number=2):
    """
    There are three modes for graph construction:
    1) using PMI (requires an appropriate PMI_FILE);
    2) using co-occurrence frequencies;
    3) unweighted graph.
    Mode number 2 is recommended.
    """
    edgeWeights = collections.defaultdict(lambda: collections.Counter())
    titles = []
    no_pmis = set()
    for title in titles_src:
        tokens = word_tokenize(title.lower())
        words = [w for w in to_words(tokens)]
        titles.append(words)
        words = [w for w in map(normalize, words) if w not in STOP_WORDS]
        for bigram in find_bigrams(words):
            bigram = tuple(sorted(bigram))
            if mode_number == 2:
                edgeWeights[bigram[0]][bigram[1]] += 1
            elif mode_number == 3:
                edgeWeights[bigram[0]][bigram[1]] = 1
            elif mode_number == 1:
                pmi = bigrams_pmis.get(bigram)
                if pmi is not None and pmi > 5:
                    edgeWeights[bigram[0]][bigram[1]] += pmi
                else:
                    no_pmis.add(bigram)
    return edgeWeights


def bigrams_to_file(uniq_bigrams, path):
    with codecs.open('bigrams_from_titles.tsv', 'w', 'utf8') as bigrams_file:
        for bigram in uniq_bigrams:
            bigrams_file.write('\t'.join(bigram))
            bigrams_file.write('\n')


##  from get_phrases.ipynb
##  should be a seperate class

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def get_phrases(words):
    if len(words) == 1:
        return words
    bigrams = find_ngrams(words, 2)
    phrases = []

    for bigram in bigrams:
        normalised = None
        p = [morph.parse(b.lower())[0] for b in bigram]
        # N + N2, N + N5
        if {'NOUN'} in p[0].tag:
            if {'NOUN', 'gent'} in p[1].tag or {'NOUN', 'ablt'} in p[1].tag:
                normalised = p[0].inflect({'nomn'}).word + ' ' + bigram[1]
                phrases.append(normalised)
            if {'NOUN', 'ablt'} in p[1].tag:
                print(p[0].inflect({'nomn'}).word + ' ' + bigram[1])

        # Adj + N
        elif {'ADJF'} in p[0].tag and {'NOUN'} in p[1].tag:
            if p[0].tag.case == p[1].tag.case and p[0].tag.number == p[1].tag.number and p[0].tag.gender == p[
                1].tag.gender:
                normalised = p[0].inflect({'nomn'}).word + ' ' + p[1].inflect({'nomn'}).word
                phrases.append(normalised)
        # Participle + N
        elif {'PRTF'} in p[0].tag and {'NOUN'} in p[1].tag:
            if p[0].tag.case == p[1].tag.case and p[0].tag.number == p[1].tag.number and p[0].tag.gender == p[
                1].tag.gender:
                normalised = p[0].inflect({'nomn'}).word + ' ' + p[1].inflect({'nomn'}).word
        # N1
        elif {'NOUN'} in p[0].tag and {'NOUN', 'gent'} not in p[1].tag and 'PREP' not in p[1].tag:
            normalised = p[0].normal_form
        elif len(p) > 1 and {'NOUN'} in p[1].tag and {'ADJF'} not in p[0].tag:
            normalised = p[1].normal_form
        # Adv + V
        elif {'ADVB'} in p[0].tag and {'VERB'} in p[1].tag:
            normalised = p[0].normal_form + ' ' + p[1].normal_form
            # Adv + V
        elif {'ADJS'} in p[0].tag and {'VERB'} in p[1].tag:
            normalised = p[0].normal_form + ' ' + p[1].normal_form
        if normalised:
            phrases.append(normalised.strip())
        # V
        if {'VERB'} in p[0].tag:
            normalised = p[0].normal_form
            phrases.append(normalised.strip())
        if {'VERB'} in p[1].tag:
            normalised = p[1].normal_form
            phrases.append(normalised.strip())

    trigrams = find_ngrams(words, 3)
    for trigram in trigrams:
        p = [morph.parse(t)[0] for t in trigram]
        normalised = None
        # N + Prep + N
        if 'NOUN' in p[0].tag and 'PREP' in p[1].tag and 'NOUN' in p[2].tag:
            normalised = p[0].inflect({'nomn'}).word + ' ' + trigram[1] + ' ' + trigram[2]
        # Adv + Adj + N
        elif 'ADVB' in p[0].tag and 'ADJF' in p[1].tag and 'NOUN' in p[2].tag:
            normalised = trigram[0] + ' ' + p[1].inflect({'nomn'}).word + ' ' + p[2].normal_form
        # Adv + Participle + N
        elif 'ADVB' in p[0].tag and 'PRTF' in p[1].tag and 'NOUN' in p[2].tag:
            normalised = trigram[0] + ' ' + p[1].inflect({'nomn'}).word + ' ' + p[2].normal_form
        # Adj + Adj + N
        elif 'ADJF' in p[0].tag and 'ADJF' in p[1].tag and 'NOUN' in p[2].tag:
            if p[0].tag.case == p[2].tag.case == p[2].tag.case and p[0].tag.gender == p[2].tag.gender and p[
                0].tag.number == p[2].tag.number:
                normalised = p[0].inflect({'nomn'}).word + ' ' + p[1].inflect({'nomn'}).word + ' ' + p[2].normal_form
        # N + Conj + N
        elif 'NOUN' in p[0].tag and 'CONJ' in p[1].tag and 'NOUN' in p[2].tag:
            normalised = p[0].inflect({'nomn'}).word + ' ' + trigram[1] + ' ' + p[2].inflect({'nomn'}).word
        # N + gent + N2
        elif {'NOUN'} in p[0].tag and {'gent'} in p[1].tag and {'NOUN', 'gent'} in p[2].tag:
            normalised = p[0].inflect({'nomn'}).word + ' ' + trigram[1] + ' ' + trigram[2]
        if normalised:
            phrases.append(normalised.strip())
    additional_phrases = []
    for idx, p in enumerate(phrases):
        if idx + 1 < len(phrases):
            next_word = morph.parse(phrases[idx + 1].split()[0])[0].normal_form
            current_word = morph.parse(p.split()[-1])[0].normal_form
            if current_word == next_word and not set(phrases[idx + 1]).issubset(set(p)):
                new_phrase = p + ' ' + ' '.join(phrases[idx + 1].split()[1:])
                additional_phrases.append(new_phrase.strip())
    if additional_phrases:
        phrases.extend(list(set(additional_phrases)))
    phrases = list(set([phrase.strip() for phrase in phrases]))
    return phrases


##  from pagerank.ipynb
##  should be a seperate Python class


def __extractNodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes


def __makeSquare(matrix, keys, default=0.0):
    matrix = matrix.copy()

    def insertMissingColumns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insertMissingColumns(matrix)  # insert missing columns
    matrix = insertMissingColumns(matrix.T).T  # insert missing rows

    return matrix.fillna(default)


def __ensureRowsPositive(matrix):
    matrix = matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(numpy.ones(len(matrix[colKey])), index=matrix.index)
    return matrix.T


def __normalizeRows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclideanNorm(series):
    return math.sqrt(series.dot(series))


def __startState(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    startProb = 1.0 / float(len(nodes))
    return pandas.Series({node: startProb for node in nodes})


def __integrateRandomSurfer(nodes, transitionProbs, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transitionProbs.copy().multiply(1.0 - rsp) + alpha


def powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000):
    # Clerical work:
    transitionWeights = pandas.DataFrame(transitionWeights)
    nodes = __extractNodes(transitionWeights)
    transitionWeights = __makeSquare(transitionWeights, nodes, default=0.0)
    transitionWeights = __ensureRowsPositive(transitionWeights)

    # Setup:
    state = __startState(nodes)
    transitionProbs = __normalizeRows(transitionWeights)
    transitionProbs = __integrateRandomSurfer(nodes, transitionProbs, rsp)

    # Power iteration:
    for iteration in range(maxIterations):
        oldState = state.copy()
        state = state.dot(transitionProbs)
        delta = state - oldState
        if __euclideanNorm(delta) < epsilon: break

    return state

def calculate_and_find_phrases(query, edgeWeights, raw_titles, mode_letter='A'):
    """
    There are four modes for ranking:
    A) simply the sum of the values;
    B) normalizing the values (by length);
    C) accounting for the topic words;
    Mode A is recommended.
    """
    query_words = query.split(' ')
    wordProbabilities = powerIteration(edgeWeights, rsp=0.15)
    wordProbabilities.sort_values(inplace=True, ascending=False)
    wordProbabilities = wordProbabilities.to_dict()
    rank2phrase = []
    for raw_title in raw_titles:
        for phrase in get_phrases(raw_title):
            query_words_bonus = 0
            if mode_letter == 'C':
                query_words_bonus = 1
            i = 1
            sum = 0
            for w in phrase.split():
                w = normalize(w)
                if w not in STOP_WORDS:
                    tr = wordProbabilities.get(w)
                    if tr:
                        sum += tr
                        i += 1
                        if mode_letter == 'C' and w in query_words:
                            query_words_bonus += 1 / ((query_words.index(w) + 1))
            if mode_letter == 'A':
                r2t = (sum, phrase)
            elif mode_letter == 'B':
                r2t = (sum / i, phrase)
            elif mode_letter == 'C':
                r2t = (sum * query_words_bonus, phrase)
            rank2phrase.append(r2t)
    rank2phrase = sorted(set(rank2phrase), reverse=True)
    return rank2phrase


def show_ranks(first_n=10):
    # pmis = get_pmi_for_bigrams()
    pmis = []
    for query, titles_src in calculate_query2titles():
        logging.info('QUERY: ' + query)
        graph = calculate_graph(query, titles_src, pmis)
        rank2phrases = calculate_and_find_phrases(query, graph, get_raw_titles(titles_src))
        if len(rank2phrases) > first_n:
            rank2phrases = rank2phrases[:first_n]
        for r2t in rank2phrases:
            print(r2t[1] + '\t' + str(r2t[0]))


def save_top_ranks(path, first_n=10):
    pmis = []
    with codecs.open(path, 'w', 'utf8') as output_file:
        for query, titles_src in calculate_query2titles():
            logging.info('==============================')
            logging.info('calculating the query: ' + str(query))
            output_file.write('TOPIC: ' + str(query))
            graph = calculate_graph(str(query), titles_src, pmis)
            rank2phrases = calculate_and_find_phrases(str(query), graph, get_raw_titles(titles_src), 'A')
            if len(rank2phrases) > first_n:
                rank2phrases = rank2phrases[:first_n]
#             result = ', '.join([str(x[1].encode("utf-8")) for x in rank2phrases]).decode("utf-8")
            result = ', '.join([str(x[1]) for x in rank2phrases])
            logging.info('result: ' + result)
            output_file.write('LABELS: ' + result + '\n\n\n')
    logging.info('saved to: ' + path)


def get_top_labels():
    pmis = []
    top_labels = []
    for query, titles_src in calculate_query2titles():
        graph = calculate_graph(query, titles_src, pmis)
        rank2phrases = calculate_and_find_phrases(query, graph, get_raw_titles(titles_src))
        top_labels.append(rank2phrases[0][1])
    return top_labels


files = [f'topics_forcomments/topic_{n}.txt' for n in range(len(os.listdir('topics_forcomments')) - 1)]
to_path = 'finished_topics/'

if __name__ == '__main__':
    # show_ranks(1)
    STOP_WORDS = 'stopwords.txt'
    # print the top label for each topic
    for number, file in enumerate(files):
        TITLES_FILE = file  ## the input file
        new_file = f'{to_path}topics_{number}.txt'  ## file to which we will save the result

        with open(new_file, 'w', encoding='windows-1251') as d:
            save_top_ranks(new_file, 5)  # save the top n labels to the specified path


## Do in terminal^
## !type *.txt > all_topics.txt