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


class User(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.Text, nullable=False)



class History(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
	vocab_id = db.Column(db.Integer, db.ForeignKey('vocabulary.id'))
	# number of times the user has been shown that word
	n_seen = db.Column(db.Float)
	# number of times the user has been tested that word
	n_test = db.Column(db.Float)
	# number of times the user get the word correctly within the test
	n_correct = db.Column(db.Float)
