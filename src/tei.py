import sys
import os
import numpy as np
import re
import argparse
import json
import glob

from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from keras_bert.datasets import get_pretrained, PretrainedList
import tensorflow as tf
from tensorflow.keras.models import load_model


BERTIMBAU_MODEL_PATH = 'models/BERTimbau/'
EMBEDDING_ID = 'sum_all_12'



def tokenize_and_compose(text):
        tokens = tokenizer.tokenize(text)
        text_tokens = []
        for i, token in enumerate(tokens):
            split_token = token.split("##")
            if len(split_token) > 1:
                token = split_token[1]
                text_tokens[-1] += token
            else:
                text_tokens.append(token)
        if len(text_tokens[1:-1]) == 1:
          return text_tokens[1]
        else:
          return text_tokens[1:-1]


def compose_token_embeddings(sentence, tokenized_text, embeddings):
        tokens_indices_composed = [0] * len(tokenized_text)
        j = -1
        for i, x in enumerate(tokenized_text):
            if x.find('##') == -1:
                j += 1
            tokens_indices_composed[i] = j
        word_embeddings = [0] * len(set(tokens_indices_composed))
        j = 0
        for i, embedding in enumerate(embeddings):
            if j == tokens_indices_composed[i]:
                word_embeddings[j] = embedding
                j += 1
            else:
                word_embeddings[j - 1] += embedding
        return word_embeddings

def extract(text, options={'sum_all_12':True}, seq_len=512, output_layer_num=12):
        features = {k:v for (k,v) in options.items() if v}
        tokens = tokenizer.tokenize(text)
        indices, segments = tokenizer.encode(first = text, max_len = seq_len)
        predicts = model_bert.predict([np.array([indices]), np.array([segments])])[0]
        predicts = predicts[1:len(tokens)-1,:].reshape((len(tokens)-2, output_layer_num, 768))

        for (k,v) in features.items():
            if k == 'sum_all_12':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts.sum(axis=1))
            if k == 'sum_last_4':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts[:,-4:,:].sum(axis=1))
            if k == 'concat_last_4':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts[:,-4:,:].reshape((len(tokens)-2,768*4)))
            if k == 'last_hidden':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts[:,-1:,:].reshape((len(tokens)-2, 768)))
        return features



def get_sentence_original_tokens(sentence, tokens):
        token_index = 0
        started = False
        sentence_pos_tokens = []
        i = 0
        while i < len(sentence):
                if sentence[i] != ' ' and not started:
                        start = i
                        started = True
                if sentence[i] == tokens[token_index] and started:
                        sentence_pos_tokens.append(sentence[i])
                        started = False
                        token_index += 1
                elif i<len(sentence) and (sentence[i] == ' ' or tokenize_and_compose(sentence[start:i+1]) == tokens[token_index] ) and started:
                        sentence_pos_tokens.append(sentence[start:i+1])
                        start = i+1
                        started = False
                        token_index += 1
                i += 1
        return sentence_pos_tokens


def get_text_location(text, arg, start_search_at=0):
    text = text.lower()
    arg = arg.lower()
    pattern = re.compile(r'\b%s\b' % arg)
    match = pattern.search(text, start_search_at)
    if match:
        return (match.start(), match.end())
    else:
        return (-1, -1)


def predict_events(text, feature_option, is_pprint=True):
    text_tokens = get_sentence_original_tokens(text, tokenize_and_compose(text))
    features = extract(text, {feature_option:True})[feature_option]
    embedding = np.array(features).reshape((len(text_tokens), 768))
    prediction = [model.predict(e.reshape((1, 768))) for e in embedding ]
    positions = list(filter((lambda i: i>= 0 and i < len(text_tokens)), [pos if pred_value > 0.5 else -1 for (pos, pred_value) in enumerate(prediction)]))
    output = []
    if len(positions) > 0:
        start_at = sum([len(token) for token in text_tokens[:positions[0]]])
    for pos in positions:
        loc_start, loc_end = get_text_location(text, text_tokens[pos], start_at)
        start_at = loc_end
        output.append({'text':  text[loc_start:loc_end],
                       'start': loc_start,
                       'end':   loc_end})
    if is_pprint:
        return json.dumps(output, indent=4)
    return output





def load_bertimbau_model():    
    global tokenizer
    global model_bert
        
    paths = get_checkpoint_paths(BERTIMBAU_MODEL_PATH)

    model_bert = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=512, output_layer_num=12)

    token_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(token_dict)


def load_tei_model():
    global model

    model = load_model('models/tiff.h5')
    return model



def identify_from_files(input_path, output_path):
    for filepathname in glob.glob(f'{input_path}*.txt'):
        extractions = []
        for line in open(filepathname):
            line = line.strip()
            print(line)
            extractions.append(predict_events(line, EMBEDDING_ID))
        filename = filepathname.split('.txt')[0].split(os.sep)[-1]
        with open(f'{output_path}{filename}.json', 'w')  as outfile:
            json.dump(extractions, outfile)
        print(f'{filename}')


def identify_events_from(input_path, output_path):
    run_identification_context(lambda : identify_from_files(input_path, output_path))
        

def identify_events_from_sentence(sentence):
    sentence = sentence.strip()
    run_identification_context(lambda : print(predict_events(sentence, EMBEDDING_ID)))
        

def run_identification_context(run_identification_func):                        
    if len(tf.config.list_physical_devices('GPU')) > 0:
        with tf.device('/GPU:0'):
            load_bertimbau_model()
            load_tei_model()
            run_identification_func()
    else:
        with tf.device("/device:CPU:0"):
            load_bertimbau_model()
            load_tei_model()
            run_identification_func()
    




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="TEI - TimeBankPT Event Identification")
    parser.add_argument('--sentence', type=str,
                        help='sentence string to identify the events')
    parser.add_argument('--dir',
                        nargs=2,
                        help='relative path to directory with files of sentences to identify events')

        

    args = parser.parse_args()

    if args.dir:
        input_dir, output_dir = args.dir
        if input_dir and output_dir and os.path.exists(input_dir) and os.path.exists(output_dir):
            identify_events_from(input_dir, output_dir)
    if args.sentence:
        identify_events_from_sentence(args.sentence)
    

    

                        
