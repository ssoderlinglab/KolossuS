
import os
import sys
import shutil
import pathlib

import tempfile
import uuid
import atexit           
import traceback 

import h5py
from io import StringIO

import numpy as np

import torch 
import torch.utils.data

import esm 
from esm import FastaBatchedDataset, pretrained
from transformers import AutoTokenizer, AutoModel

# for handling padding characters 
from .read_data import PADDING_CHAR

# warnings for testing 
from .warnings import warn

# for writing fasta 
from .write_fasta import write_fasta

# constants 
ESM_TRANSFORMER_LAYERS = {
        "esm2_t6_8M_UR50D": 6,
        "esm2_t12_35M_UR50D": 12,
        "esm2_t30_150M_UR50D": 30,
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t48_15B_UR50D": 48
}


# FLAG 274: for Kanchan to test the code and make sure it's functional
# NOTE: remove testing in final version
# seq_list is a list of tuples (seq_id, seq)
@warn(274)
def get_embeddings(seq_list, device, model_name, testing=False):
    if testing:
        return _get_embeddings_testing(seq_list)
    
    # print("ERROR: UNTESTED CODE: KANCHAN PLEASE TEST THIS OUT IN FILE", __file__, "!!!", file=sys.stderr)
    # sys.exit(1)
    
    # again, for Kanchan to test this out and debug
    # Load ESM-2 model
    model, alphabet = load_model()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [(a, s.strip(PADDING_CHAR)) for a, s in seq_list]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[48], return_contacts=True)
    token_representations = results["representations"][48]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, ((seq_id, _), tokens_len) in enumerate(zip(seq_list, batch_lens)):
        seq_rep = token_representations[i, 1 : tokens_len - 1].mean(0)
        sequence_representations.append((seq_id, seq_rep))

    return sequence_representations
    

def _get_embeddings_testing(seq_list):
    d = {'A': 0, 'C': 1, 'G': 2, 'T': 3, PADDING_CHAR: 4}
    out = [None for _ in range(len(seq_list))]
    dim = 5120
    for i, (seq_id, s) in enumerate(seq_list):
        x = torch.Tensor([d[c] for c in s] + [-100 for _ in range(dim-len(s))])
        out[i] = (seq_id, x)
    
    return out


def extract_embeddings(seq_list, device, model_name='esm2_t48_15B_UR50D', output_dir=None, 
                       tokens_per_batch=4096, seq_length=1022, repr_layers=None, 
                       layer_to_use=None, keep_temp=False):
    # Determine layers
    if repr_layers is None or layer_to_use is None:
        try:
            layer = ESM_TRANSFORMER_LAYERS[model_name]
            repr_layers = repr_layers or [layer]
            layer_to_use = layer_to_use or layer
        except KeyError as e:
            raise ValueError(f"Unknown model name: {model_name}") from e

    # Handle output_dir and tempdir
    if output_dir is None:
        output_dir = pathlib.Path(f'./{uuid.uuid4().hex}')
        tempdir = output_dir
    else:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tempdir = output_dir / f'tmp_{uuid.uuid4().hex}'

    tempdir.mkdir(parents=True, exist_ok=True)
    if not keep_temp:
        atexit.register(lambda: shutil.rmtree(tempdir, ignore_errors=True))

    print('-' * 40)
    print("Loading ESM model:", model_name)
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model.to(device if torch.cuda.is_available() else 'cpu')
    print("Model loaded.")

    try:
        fasta_file = tempdir / "seqs.fasta"
        write_fasta([(t[0], t[1].strip(PADDING_CHAR)) for t in seq_list], fasta_file)
        nseqs = len(seq_list)
        del seq_list

        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(seq_length),
            batch_sampler=batches
        )

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(f'Processing batch {batch_idx + 1} of {len(batches)}')
                toks = toks.to(device=device, non_blocking=True) if torch.cuda.is_available() else toks

                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {layer: t.cpu() for layer, t in out["representations"].items()}

                for i, label in enumerate(labels):
                    entry_id = label.split()[0]
                    truncate_len = min(seq_length, len(strs[i]))

                    result = {
                        "entry_id": entry_id,
                        "mean_representations": {
                            layer: t[i, 1:truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items()
                        }
                    }
                    torch.save(tempdir / f"{entry_id}.pt", result)

        print("Extracting mean representations of sequences")
        out = {}
        for fname in tempdir.glob("*.pt"):
            sid = fname.stem
            sx = torch.load(fname)['mean_representations'][layer_to_use]
            out[sid] = sx

        print(f"Done converting {len(out)} / {nseqs} sequences to embeddings")
        print(f"Closing temporary directory at {tempdir}")
        print('-'*40)

        return out

    except Exception as e:
        print("Error encountered while creating sequence embeddings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise e


def extract_embeddings_from_fasta(fasta_file, output_file, device, model_name, 
                       tokens_per_batch=4096, seq_length=1022,repr_layers=None, layer_to_use=None):
    # handle repr_layers 
    if repr_layers is None: 
        try:
            repr_layers = [ESM_TRANSFORMER_LAYERS[model_name]]
        except KeyError as e:
            raise e
            
    # handle layer_to_use
    if layer_to_use is None: 
        try:
            layer_to_use = ESM_TRANSFORMER_LAYERS[model_name]
        except KeyError as e:
            raise e
            
    # assert valid output and input files 
    assert os.path.isfile(fasta_file)
    assert os.path.isdir(os.path.join(*os.path.split(output_file)[:-1]))
    if not output_file.endswith('.h5'):
        output_file += '.h5'

    print('-' * 40)
    print("Loading esm model")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    print("Esm model done loading")
    model.eval()

    if torch.cuda.is_available():
        model.to(device)
    else:
        model.to('cpu')

    dataset = FastaBatchedDataset.from_file(pathlib.Path(fasta_file))
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches)

    with torch.no_grad(): 
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
    
            print(f'Processing batch {batch_idx + 1} of {len(batches)}')
    
            if torch.cuda.is_available():
                toks = toks.to(device=device, non_blocking=True)
    
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
    
            with h5py.File(output_file, 'a') as fout:
                for i, label in enumerate(labels):
                    entry_id = label
                    truncate_len = min(seq_length, len(strs[i]))
    
                    key = entry_id
                    mean_representations = {
                            layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items() }
    
                    fout[key] = mean_representations[layer_to_use]

    print("saved all sequences to", output_file)


def extract_embeddings_from_fasta_individual(fasta_file, output_dir, device, model_name, 
                       tokens_per_batch=4096, seq_length=1022,repr_layers=None,layer_to_use=None):
    # handle repr_layers 
    if repr_layers is None: 
        try:
            repr_layers = [ESM_TRANSFORMER_LAYERS[model_name]]
        except KeyError as e:
            raise e
            
    # handle layer_to_use
    if layer_to_use is None: 
        try:
            layer_to_use = ESM_TRANSFORMER_LAYERS[model_name]
        except KeyError as e:
            raise e
            
    # assert valid output and input files 
    assert os.path.isfile(fasta_file)
    assert os.path.isdir(output_dir)

    print('-' * 40)
    print("Loading esm model")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    print("Esm model done loading")
    model.eval()

    if torch.cuda.is_available():
        model.to(device)
    else:
        model.to('cpu')

    dataset = FastaBatchedDataset.from_file(pathlib.Path(fasta_file))
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches)

    with torch.no_grad(): 
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
    
            print(f'Processing batch {batch_idx + 1} of {len(batches)}')
    
            if torch.cuda.is_available():
                toks = toks.to(device=device, non_blocking=True)
    
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
    
            for i, label in enumerate(labels):
                output_file = f'{output_dir}/{label}.h5'
                with h5py.File(output_file, 'w') as fout:
                    entry_id = label
                    truncate_len = min(seq_length, len(strs[i]))
    
                    key = entry_id
                    mean_representations = {
                            layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items() }
    
                    fout[key] = mean_representations[layer_to_use]

    print("saved all sequences to", output_dir)


# in case we need to modify later
def load_model():
    return esm.pretrained.esm2_t48_15B_UR50D()

