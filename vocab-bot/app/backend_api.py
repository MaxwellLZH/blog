from sqlalchemy.ext.declarative import DeclarativeMeta
import json
import random

from app import db
from app.models import Vocabulary, User, History



class AlchemyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data) # this will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)



def get_n_words(category: str, n: int=20):
	words = Vocabulary.query.filter_by(category=category).limit(n).all()
	return json.dumps([w.to_dict() for w in words])



