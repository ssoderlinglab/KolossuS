
from .kolossus import BATCH_SIZE
from .kolossus import set_batch_size
from .kolossus import kolossus

import argparse
import sys 
import h5py


def main():
    args = parse_args()

    print("flag 1: help!!!!", __file__)

    if args['batch_size']:
        print(f"setting batch size to {args['batch_size']}...", end=' ')
        set_batch_size(args['batch_size'])
        print("done!")

    print("running model...", end=' ')
    results = kolossus(**args['kolossus'])
    print("done!")

    output_results(results, **args['output'])


def output_results(results, dists_fname_out, projections_fname_out):
    if projections_fname_out:
        results, projections = results
        print("writing embeddings to:", projections_fname_out)
        with h5py.File(projections_fname_out, 'w') as fout:
            for _id, e in projections.items():
                fout[_id] = e

    print("writing predictions to", dists_fname_out)
    with open(dists_fname_out, 'wt') as fout:
        print('#kinase\tsubstrate\tpredicted_prob', file=fout)
        for pair, prob in results.items():
            pair = tuple(map(str, pair))
            prob = str(prob)
            print('\t'.join((*pair, prob)), file=fout)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pairs', type=str, required=True, 
                        help="format: <kinase_id>\t<substrate_id>\t<substrate_phosphorylation_site>")
    parser.add_argument('-s', '--seqs', type=str, default=None, 
                        help='fasta formatted file of sequences, either seqs or embeddings must be provided')
    parser.add_argument('-e', '--embeddings', type=str, default=None, 
                        help='h5 file of sequence embeddings, either seqs or embeddings must be provided')
    parser.add_argument('--dtype', type=str, default='', 
                        help='data type of sequence embeddings (usually float32)')
    parser.add_argument('-r', '--projections', type=str, default='', 
                        help='name of .h5 files for kolossus projections')
    parser.add_argument('-d', '--device', type=str, default='cpu', 
                        help='default device on which to run model')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Number of pairs at a time on which to run model')
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help='desired file path for output')
    parser.add_argument('--model-small', dest='model_small', action='store_true', 
                        help='flag to use the KolossuS model that uses ESM2-650M parameters')
    args = parser.parse_args()

    real_args = {'kolossus': {'fpairs': args.pairs,
                         'fseqs': args.seqs,
                         'fembeds': args.embeddings,
                         'dtype': args.dtype,
                         'device': int(args.device) if args.device.isdigit() else args.device,
                         'return_projections': len(args.projections) > 0,
                         'model': 'small' if args.model_small else 'large'},
                 'batch_size': args.batch_size,
                 'output': {'dists_fname_out': args.output,
                            'projections_fname_out': args.projections if len(args.projections) > 0 else None}}
    
    return real_args    


if __name__ == '__main__':
    main()
