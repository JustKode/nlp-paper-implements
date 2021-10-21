from attention import Attention
from data import make_data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


sentences = [
    ("나는 너를 사랑해", "i love you"),
    ("너는 나를 좋아해", "you like me"),
    ("그들은 치킨을 먹어", "they eat chicken"),
    ("내 친구들은 피자를 먹어", "my friends eat pizza"),
    ("그녀는 콜라를 좋아해", "she like coke")
]

(inputs, input_word_list_len, input_word_to_idx, input_idx_to_word, input_longest_sequence),\
    (outputs, output_word_list_len, output_word_to_idx, output_idx_to_word, output_longest_sequence),\
    force_teachers = make_data(sentences)

if __name__ == "__main__":
    torch.manual_seed(0)
    print(input_idx_to_word)
    print(output_idx_to_word)
    BATCH_SIZE = len(sentences)
    N_HIDDEN = 128

    model = Attention(input_word_list_len, output_word_list_len, batch_size=BATCH_SIZE, n_hidden=N_HIDDEN)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.002)

    # training
    for epoch in range(2000):
        optimizer.zero_grad()
        output = model(inputs, force_teachers)
        loss = criterion(output.view(-1, output_word_list_len), outputs.view(-1))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Cost={loss:.6f}")

        loss.backward()
        optimizer.step()

    # evaluating
    test_batch = [
        [j for j in [output_word_to_idx["<SOS>"]] + [output_word_to_idx["<POS>"] for j in range(output_longest_sequence - 1)]] for i in range(BATCH_SIZE)
    ]

    test_batch = torch.LongTensor(test_batch)
    output = model(inputs, test_batch)

    print(inputs)
    print(outputs)
    print(test_batch)

    output = list(output.argmax(dim=2).numpy())
    print(output)
    for i, line in enumerate(output):
        print(f"{sentences[i][0]} => {' '.join(map(lambda x: output_idx_to_word[x], line))}")
    print(output)