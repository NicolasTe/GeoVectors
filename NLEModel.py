import pickle
import pandas as pd
import numpy as np
import gc
import ntpath
import io

from os.path import join, abspath, exists
from os import makedirs
from math import isnan
from nodevectors import Node2Vec
from WeightedDeepWalkGraph import _edgelist_to_wdw_graph
from BaseModel import BaseModel
from tqdm import tqdm
from haversine import haversine

from joblib import Parallel, delayed, parallel_backend

"""
This is class represents the neural location embedding model for OpenStreetMap entities.
"""


def get_key_x_y(r):
    key = r['type'] + "_" + str(r['id'])
    x = r['lat']
    y = r['lon']
    return key, x, y


def get_np_key_x_y(r):
    key = r[2]+"_"+str(r[1])
    x = r[4]
    y = r[5]
    return key, x, y


def get_dump_key_x_y(r):
    key = r[1]+"_"+str(r[0])
    x = r[3]
    y = r[4]
    return key, x, y


def sliding_window_slice(np_data, i, rows, num_rows, pbar):
    r = np_data[i]
    key, x, y = get_np_key_x_y(r)

    if isnan(x) or isnan(y):
        return

    window_size = min(num_rows - i, 51)
    for j in range(1, window_size):
        o_key, o_x, o_y = get_np_key_x_y(np_data[i + j])

        if isnan(o_x) or isnan(o_y):
            continue

        dist = haversine((x, y), (o_x, o_y))

        rows.append((key, o_key, dist))
    pbar.update(1)


def apply_sliding_window(data, run_number, njobs, rows):
    np_data = data.to_numpy()

    num_rows = np_data.shape[0]
    pbar = tqdm(total=num_rows, desc="Applying sliding window "+str(run_number))
    with parallel_backend('threading', n_jobs=njobs):
        results = Parallel()(delayed(sliding_window_slice)(np_data, i, rows, num_rows, pbar) for i in range(num_rows))

    pbar.close()
    return rows


class NLEModel(BaseModel):

    def __init__(self, index_path, njobs,  db):
        BaseModel.__init__(self)
        self.index_path = abspath(index_path)
        if not exists(self.index_path):
            makedirs(self.index_path)

        self.njobs = njobs
        self.idx = None
        self.wdw = None
        self.db = db
        self.table_name = "geovectors_nle"

    def train(self, data):
        # index data
        self._setup_db()

        copy_input = io.StringIO()
        error_counter = 0
        for i, r in enumerate(tqdm(data.iterrows(), total=len(data.index), desc="Building index")):
            row = []
            key, x, y = get_key_x_y(r[1])

            row.append(key)

            if isnan(x) or isnan(y):
                error_counter += 1
                continue

            row.append("SRID=4326;POINT(" + str(x) + " " + str(y) + ")")
            copy_input.write(",".join(row) + "\n")

        if error_counter > 0:
            print("Warning could not encode " + str(error_counter) + " instances")

        print("Executing copy to db")
        copy_input.seek(0)

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.copy_from(copy_input, self._table_name(), sep=",", null="\"\"")

                print("Creating index")
                idx_query = "create index "+self._table_name()+"_loc_idx on "+self._table_name()+" using gist(location);"
                cur.execute(idx_query)
        print("Indexing done")
        return None


        rows = []
        data.sort_values("lat", inplace=True)
        apply_sliding_window(data, 1, self.njobs, rows)

        data.sort_values("lon", inplace=True)
        apply_sliding_window(data, 2, self.njobs, rows)

        print("Created " + str(len(rows))+" edges")

        elist = pd.DataFrame(rows, columns=["src", "dst", "weight"])
        elist.weight = pd.to_numeric(elist.weight)

        # Create name mapping to normalize node IDs
        allnodes = list(set(elist.src.unique()).union(set(elist.dst.unique())))

        # This factors all the unique nodes to unique IDs
        names = (
            np.array(
                pd.Series(allnodes).astype('category')
                    .cat.categories
            ))
        name_dict = dict(zip(names,
                             np.arange(names.shape[0])))

        elist.src = elist.src.map(name_dict).astype(np.uint32)
        elist.dst = elist.dst.map(name_dict).astype(np.uint32)
        elist.sort_values(by='src', inplace=True, ignore_index=True)

        nnodes = names.shape[0]
        G = _edgelist_to_wdw_graph(elist, nnodes, nodenames=names)

        elist = None
        rows = None
        gc.collect()

        # train node2vec
        print("Training node2vec")

        wdw = Node2Vec(threads=self.njobs, walklen=10, n_components=100)
        wdw.fit(G)

        print("Training complete")

        self.wdw = wdw

    def encode_pandas_instance(self, instance):
        key, x, y = get_key_x_y(instance[1])
        return self.encode_coords(key, x,y)

    def encode_instance(self, instance):
        key, x, y = get_dump_key_x_y(instance)
        return self.encode_coords(key, x, y)

    def encode_coords(self, key, x,y):
        if isnan(x) or isnan(y):
            return None
        else:
            vectors = []
            nearest_neighbors = self._get_nn(x, y, 50)

            dist_sum = 0
            for n in nearest_neighbors:
                #other_key = n.object
                other_key = n[0]

                if key == other_key:
                    continue

                #use distance in kilometers
                dist = np.log(1 + (1 / (n[1] / 1000)))
                dist_sum += dist

                node_enc = self.wdw.predict(other_key)
                vectors.append(node_enc * dist)

            enc = np.sum(vectors, axis=0) / dist_sum
            return enc
    def _get_nn(self, x, y, n):
        result = []
        point = "public.st_setsrid(public.st_makepoint("+str(x)+", "+str(y)+"), 4326)"
        nn_query =  "select osm_key, " +\
                    " public.st_distance(location::public.geography, " +\
                    point+"::public.geography) " +\
                    " from "+self._table_name()+" " +\
                    " where public.st_distance(location::public.geography, "+point+"::public.geography) > 0 " +\
                    " order by location OPERATOR(public.<->) "+point+" " +\
                    " limit "+str(n)+"; "

        try:
            conn = self.db.get_pool_connection()
            with conn.cursor() as cur:
                cur.execute(nn_query)
                rows = cur.fetchall()
                for r in rows:
                    result.append((r[0], r[1]))

            self.db.free_pool_connection(conn)
        except:
            print("Query")
            print(x, y,)
            print(nn_query)
            print("_____________")
            raise

        return result

    def _setup_db(self):
        drp_query = "drop table if exists "+self._table_name()+";"

        query = "create table "+self._table_name()+" ( " +\
                "osm_key text primary key, " +\
                "location public.geometry(Point, 4326));"

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(drp_query)
                cur.execute(query)

    def _table_name(self):
        return ntpath.split(self.table_name)[1].replace(".", "_")

    def _wdw_path(self):
        return join(self.index_path, "wdw.pickle")

    def load_indexes(self):
        self.db.create_connection_pool()

        with open(self._wdw_path(), 'rb') as fi:
            self.wdw = pickle.load(fi)

        super()._set_dimensions(self.wdw.n_components)

    def save_model(self, path):

        with open(self._wdw_path(), 'wb') as fo:
            pickle.dump(self.wdw, fo)

    def destroy(self):
        self.db.close_connection_pool()
