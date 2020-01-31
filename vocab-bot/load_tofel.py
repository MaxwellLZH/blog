import tqdm
from app import db
from app.models import Vocabulary


def remove_multiple_space(s):
    return ' '.join(s.split())


with open('./data/tofel.txt', encoding='utf8') as f:
    lines = [remove_multiple_space(l) for l in f.readlines()]
    for i, line in tqdm.tqdm(enumerate(lines)):
        word, phonetic, *meaning = line.split()
        meaning = ' '.join(meaning)
        category = 'TOFEL'

        w = Vocabulary(category=category, word=word, phonetic=phonetic, meaning=meaning)    
        db.session.add(w)

        if i % 50 == 0:
            db.session.commit()

