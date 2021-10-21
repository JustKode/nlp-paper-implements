import numpy as np
import torch

def make_data(sentences):
    inputs = []
    input_word_list = set()
    input_longest_sequence = 0
    outputs = []
    output_word_list = set()
    output_longest_sequence = 0
    force_trainers = []

    for input, output in sentences:
        input_words = input.split()
        for word in input_words:
            input_longest_sequence = max(len(input_words), input_longest_sequence)
            input_word_list.add(word)

        output_words = output.split()
        for word in output_words:
            output_longest_sequence = max(len(output_words), output_longest_sequence)
            output_word_list.add(word)

    input_longest_sequence += 1
    output_longest_sequence += 1

    input_word_list = ['<SOS>', '<EOS>', '<POS>'] + list(input_word_list)
    output_word_list = ['<SOS>', '<EOS>', '<POS>'] + list(output_word_list)
    input_word_list_len = len(input_word_list)
    output_word_list_len = len(output_word_list)

    input_word_to_idx = {w: i for i, w in enumerate(input_word_list)}
    input_idx_to_word = {i: w for i, w in enumerate(input_word_list)}
    output_word_to_idx = {w: i for i, w in enumerate(output_word_list)}
    output_idx_to_word = {i: w for i, w in enumerate(output_word_list)}

    for input, output in sentences:
        input_vector = []
        for word in input.split():
            input_vector.append(input_word_to_idx[word])

        for i in range(input_longest_sequence - len(input_vector)):
            input_vector.append(input_word_to_idx['<POS>'])
        inputs.append(input_vector)

        output_vector = []
        force_trainer = [output_word_to_idx['<SOS>']]
        for word in output.split():
            output_vector.append(output_word_to_idx[word])
            force_trainer.append(output_word_to_idx[word])

        output_vector.append(output_word_to_idx['<EOS>'])

        for i in range(output_longest_sequence - len(output_vector)):
            output_vector.append(output_word_to_idx['<POS>'])
            force_trainer.append(output_word_to_idx['<POS>'])
        outputs.append(output_vector)
        force_trainers.append(force_trainer)

    inputs = torch.LongTensor(inputs)
    outputs = torch.LongTensor(outputs)
    force_trainers = torch.LongTensor(force_trainers)

    return (inputs, input_word_list_len, input_word_to_idx, input_idx_to_word, input_longest_sequence), \
           (outputs, output_word_list_len, output_word_to_idx, output_idx_to_word, output_longest_sequence), force_trainers