import fasttext
import fasttext.util
import ast
import numpy as np
from BaseModel import BaseModel

"""
This is class represents the Fasttext embedding model for OpenStreetMap tags.
"""

class FastTextModel(BaseModel):

    def __init__(self, model_path):
        BaseModel.__init__(self)
        self.ft = fasttext.load_model(model_path)
        self.dimension = self.ft.get_dimension()

    def train(self, data):
        pass

    def encode_pandas_instance(self, instance):
        tags = ast.literal_eval(instance[1]['tags'])
        return self.encode_tags(tags)

    def encode_instance(self, instance):
        return self.encode_tags(instance[2])

    def encode_tags(self, tags):
        if len(tags) == 0:
            return None
        else:
            vectors = []
            for k, v in tags:
                vectors.append(self.ft.get_word_vector(k))
                if len(v.split()) == 1:
                    vectors.append(self.ft.get_word_vector(v))

            enc = np.mean(vectors, axis=0)
        return enc

    def save_model(self, path):
        pass



