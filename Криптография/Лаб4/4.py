import secrets
import string
import urllib.request

MAX_TEXT_SIZE = 400000
TEXT_SIZE = 1

links = {
    'https://www.gutenberg.org/files/1497/1497.txt': 793,
    'https://www.gutenberg.org/files/37090/37090.txt': 7123,
    'https://www.gutenberg.org/files/25447/25447-0.txt': 27477,
    'https://www.gutenberg.org/files/10378/10378.txt': 1931
}

eng_words = []

b_print_texts = False


def print_texts(text_1, text_2):
    print('TEXT_1: ')
    print(text_1)
    print('TEXT_2: ')
    print(text_2)
    print()


def text_from_random_words(words):
    if not len(words):
        url = 'http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain'
        words += urllib.request.urlopen(url).read().decode().splitlines()
    random_text = ''
    while len(random_text) <= TEXT_SIZE:
        random_text += secrets.choice(words) + ' '
    random_text = random_text.strip()
    return random_text[:TEXT_SIZE]


def calculate_ratio(text_1, text_2):
    count = 0
    for i in range(len(text_1)):
        if text_1[i] == text_2[i]:
            count += 1
    return count / len(text_1)


def case_1():
    print('Два осмысленных текста на естественном языке')

    while True:
        link_1 = secrets.choice(list(links.keys()))
        link_2 = secrets.choice(list(links.keys()))
        if link_1 != link_2:
            break

    text_1 = urllib.request.urlopen(link_1).read().decode()[links[link_1]:][:TEXT_SIZE]
    text_2 = urllib.request.urlopen(link_2).read().decode()[links[link_2]:][:TEXT_SIZE]

    if b_print_texts:
        print_texts(text_1, text_2)

    print(calculate_ratio(text_1, text_2))


def case_2():
    print('Осмысленный текст и текст из случайных букв')

    link = secrets.choice(list(links.keys()))
    text = urllib.request.urlopen(link).read().decode()[links[link]:][:TEXT_SIZE]
    random_text = ''.join(secrets.choice(string.ascii_letters + ' ') for _ in range(TEXT_SIZE))

    if b_print_texts:
        print_texts(text, random_text)

    print(calculate_ratio(text, random_text))


def case_3():
    print('Осмысленный текст и текст из случайных слов')

    link = secrets.choice(list(links.keys()))
    text = urllib.request.urlopen(link).read().decode()[links[link]:][:TEXT_SIZE]
    random_text = text_from_random_words(eng_words)

    if b_print_texts:
        print_texts(text, random_text)

    print(calculate_ratio(text, random_text))


def case_4():
    print('Два текста из случайных букв')

    random_text_1 = ''.join(secrets.choice(string.ascii_letters + ' ') for _ in range(TEXT_SIZE))
    random_text_2 = ''.join(secrets.choice(string.ascii_letters + ' ') for _ in range(TEXT_SIZE))

    if b_print_texts:
        print_texts(random_text_1, random_text_2)

    print(calculate_ratio(random_text_1, random_text_2))


def case_5():
    print('Два текста из случайных слов')

    random_text_1 = text_from_random_words(eng_words)
    random_text_2 = text_from_random_words(eng_words)

    if b_print_texts:
        print_texts(random_text_1, random_text_2)

    print(calculate_ratio(random_text_1, random_text_2))


def run_all():
    case_1()
    case_2()
    case_3()
    case_4()
    case_5()


if __name__ == '__main__':
    TEXT_SIZE = int(input(f'Введите размер текста (не более {MAX_TEXT_SIZE})\n'))

    if TEXT_SIZE > MAX_TEXT_SIZE:
        raise Exception('Запрашиваемый размер слишком большой')

    if TEXT_SIZE <= 0:
        raise Exception('Некорректный размер')

    pt = input('Печать текстов: y - да, n - нет\n')
    if pt == 'y':
        b_print_texts = True

    print('1 - Два осмысленных текста на естественном языке')
    print('2 - Осмысленный текст и текст из случайных букв')
    print('3 - Осмысленный текст и текст из случайных слов')
    print('4 - Два текста из случайных букв')
    print('5 - Два текста из случайных слов')
    print('a - Все подряд')

    case = input()
    if case == '1':
        case_1()
    elif case == '2':
        case_2()
    elif case == '3':
        case_3()
    elif case == '4':
        case_4()
    elif case == '5':
        case_5()
    elif case == 'a':
        run_all()
