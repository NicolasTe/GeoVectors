from tqdm import tqdm
from AsyncWrite import AsyncWrite
from time import sleep
import numpy as np

"""
This is a base class for OSM embeddings models.
"""
class BaseModel:

    def __init__(self):
        self.n_dimensions = -1

    def _set_dimensions(self, n):
        self.n_dimensions = n

    def train(self, data):
        pass

    def encode(self, data, output):
        writer = AsyncWrite(output)
        writer.start()

        for r in tqdm(data.iterrows(), total=len(data.index)):
            type = r[1]['type']
            id = r[1]['id']
            enc = self.encode_pandas_instance(r)

            if enc is None:
                enc = np.zeros(self.n_dimensions)

            while writer.qizes() >= 1e4:
                sleep(3)

            writer.add_line([type, id] + list(enc))

        writer.set_done()
        writer.join()

    def encode_instance(self, instance):
        raise Exception("encode_instance is not implemented.")

    def save_model(self, path):
        raise Exception("save_model is not implemented.")

    def destroy(self):
        pass