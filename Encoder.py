import argparse
import util
import os.path
import itertools
from AsyncWrite import AsyncWrite
from PostgresDB import PostgresDB
from FasttextModel import FastTextModel
from NLEModel import NLEModel
from os import listdir, makedirs

"""
This is the main script to run the GeoVectors embedding framework.
Given an OpenStreetMap snapshost, this script encode the entities from the snapshot into embeddings.
Arguments:
    input               File of osm samples or directory with country dumps
    output              Directory to save encoded files
    modelPath           Path to pretrained model
    db_credentials      Credentials for the database
    model               Name of the embedding model. Either fasttext or nle
    njobs               Number of threads to use (optional)

"""
def get_model(args):
    if args.model == 'fasttext':
        model = FastTextModel(args.modelPath)
    elif args.model == 'nle':
        db = PostgresDB(util.read_db_config(args.db_cred))
        model = NLEModel(args.modelPath, args.njobs, False, db)
        model.load_indexes()
    return model


def run_on_dump(args, input, output):
    model = get_model(args)

    writer = AsyncWrite(output, compressed=True, encoder=model)
    writer.start()

    data = util.read_from_snapshot(input, writer=writer, max_runs=2)

    for record in itertools.chain.from_iterable(data):
        writer.add_line(record)

    writer.set_done()
    writer.join()
    model.destroy()


def run_dumps(args):
    # identify targets in directory
    targets = [f for f in listdir(args.input) if f.endswith(".osm.pbf")]

    if not os.path.isdir(args.output):
        makedirs(args.output)

    f_args = {}
    for t in targets:
        fname = os.path.splitext(t)[0]
        fname = os.path.splitext(fname)[0]+"_"+args.model+".tsv.gz"
        input = os.path.join(args.input, t)
        output = os.path.join(args.output, fname)
        f_args[t] = (args, input, output)

    util.run_on_snapshots(run_on_dump, f_args, targets, args.njobs)


def run():
    parser = argparse.ArgumentParser(description='Encode embeddings')

    parser.add_argument('input', help='Input: file of osm samples or directory with country dumps')
    parser.add_argument('output', help='Directory to save encoded files')
    parser.add_argument('modelPath', help='Path to pretrained model', type=str)

    parser.add_argument('--db_cred', help='Credentials for the database', type=str, default="")
    parser.add_argument('--model', help='Name of the embedding model. Either fasttext or nle', default="fasttext", type=str)
    parser.add_argument('--njobs', help='Number of threads to use', default=1, type=int)

    args = parser.parse_args()
    run_dumps(args)


if __name__ == '__main__':
    run()
