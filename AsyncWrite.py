import threading
import gzip
from multiprocessing import Manager

"""
This class provides an asynchronous writer that runs in its own thread.
"""


class AsyncWrite(threading.Thread):

    def __init__(self, fname, compressed=False, encoder=None):
        threading.Thread.__init__(self)
        self.q = Manager().Queue()
        self.done = False
        self.fname = fname
        self.compressed = compressed
        self.encoder = encoder

    def add_line(self, line):
        self.q.put(line)

    def qizes(self):
        return self.q.qsize()

    def open_file(self):
        if self.compressed:
            return gzip.open(self.fname, 'wb')
        else:
            return open(self.fname, 'w', encoding='utf-8')

    def transform_line(self, l):
        if self.encoder is not None:
            enc = self.encoder.encode_instance(l)
            if enc is None:
                return None
            l = [l[1], l[0]] + list(enc)
        return "\t".join(map(str, l))+"\n"

    def run(self):
        if self.compressed:
            file_encoding = (lambda x: x.encode("utf-8"))
        else:
            file_encoding = (lambda x: x)

        with self.open_file() as fo:
            while not self.done:
                while not self.q.empty():
                    l = self.q.get()
                    l = self.transform_line(l)
                    if l is None:
                        continue
                    l = file_encoding(l)
                    fo.write(l)

    def set_done(self):
        self.done = True
