from app import db


class Vocabulary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(20), index=True)
    word = db.Column(db.String(120), index=True)
    phonetic = db.Column(db.String(120))
    meaning = db.Column(db.String(240))

    def __repr__(self):
        return '【{}, {}, {}, {}】'.format(self.category, self.word, self.phonetic, self.meaning)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
