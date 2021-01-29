import argparse
from util import read_samples, read_db_config
from PostgresDB import PostgresDB

from FasttextModel import FastTextModel
from NLEModel import NLEModel

"""
This is script can be used to train an embedding model on a sample from OpenStreetMap.
"""

def run():
    parser = argparse.ArgumentParser(description='Calcuclate embeddings')

    parser.add_argument('input', help='Input containing osm samples')
    parser.add_argument('output', help='File to safe embeddings in')

    parser.add_argument('--model', help='Embedding model to us', default="NLE", type=str)
    parser.add_argument('--ftmodel', help='Path to fasttext model', type=str, default="")
    parser.add_argument('--db_cred', help='Credentials for database', type=str, default="")
    parser.add_argument('--njobs', help='Number of threads to use', default=1, type=int)


    args = parser.parse_args()


    # create model
    if args.model == "fasttext":
        model = FastTextModel(args.ftmodel)
    elif args.model == 'NLE':
        db = PostgresDB(read_db_config(args.db_cred))
        model = NLEModel(args.output, args.njobs, db)
    else:
        raise Exception("Model not found")

    data = read_samples(args.input)

    # train model
    model.train(data)

    # save model
    model.save_model(args.output+"_"+args.model)


if __name__ == '__main__':
    run()
