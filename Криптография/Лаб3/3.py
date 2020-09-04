import secrets
import bitarray
import matplotlib.pyplot as plt

import Keccak

MAX_NUM_OF_ROUNDS = 24
SIZE_OF_MESSAGE_BYTE = 512
SIZE_OF_ENCRYPTED_MESSAGE_BIT = 1024


def calculate_difference(msg_1, msg_2):
    count = 0

    bytes_1 = bytes.fromhex(msg_1)
    bytes_2 = bytes.fromhex(msg_2)

    ba_1 = bitarray.bitarray()
    ba_1.frombytes(bytes_1)

    ba_2 = bitarray.bitarray()
    ba_2.frombytes(bytes_2)

    for i in range(len(ba_1)):
        if ba_1[i] != ba_2[i]:
            count += 1
    return count


def change_one_bit(message):
    ba = bitarray.bitarray()
    ba.frombytes(message)
    bite_idx = secrets.randbelow(len(ba))
    ba[bite_idx] = True if ba[bite_idx] is False else False
    return ba.tobytes()


def get_random_message(length):
    return secrets.token_bytes(length)


def graph(differences):
    plt.bar([i for i in range(1, MAX_NUM_OF_ROUNDS + 1)], differences)
    plt.title(f'Размер зашифрованного сообщения {SIZE_OF_ENCRYPTED_MESSAGE_BIT} бит')
    plt.xlabel('Число раундов')
    plt.ylabel('Число изменившихся бит')
    plt.show()


def main():
    my_keccak = Keccak.Keccak()

    message_1 = get_random_message(SIZE_OF_MESSAGE_BYTE)
    message_2 = change_one_bit(message_1)

    message_1 = message_1.hex().upper()
    message_2 = message_2.hex().upper()

    assert len(message_1) == len(message_2)

    differences = []
    for num_round in range(1, MAX_NUM_OF_ROUNDS + 1):
        encrypted_message_1 = my_keccak.Keccak((len(message_1) * 4, message_1),
                                               num_of_rounds=num_round, n=SIZE_OF_ENCRYPTED_MESSAGE_BIT)
        encrypted_message_2 = my_keccak.Keccak((len(message_1) * 4, message_2),
                                               num_of_rounds=num_round, n=SIZE_OF_ENCRYPTED_MESSAGE_BIT)
        differences.append(calculate_difference(encrypted_message_1, encrypted_message_2))

    graph(differences)


if __name__ == '__main__':
    main()
