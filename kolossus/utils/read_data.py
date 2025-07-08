
import sys
from collections import defaultdict 

# for seqs
from .parse_fasta import parse_fasta 
from .seqops import extract_window

# FLAG 274: consult with Kanchan as far as how to pad 
PADDING_CHAR = '#'
WINDOW_SIZE = 15

# for embeddings
import h5py
import torch
# from .seq_to_embedding import get_embeddings
from .seq_to_embedding import extract_embeddings

# warnings for testing 
from .warnings import warn


# FLAG 274: for kanchan: how to handle padding for substrate sites near start or end of sequence 
# assume pairs are in form (kinase_sequence_id, substrate_sequence_id, substrate_phosphorylation_site), tab or comma separated
@warn(274)
def read_pairs(fname, delimiter='\t', includes_window=True):
    pairs = []
    with open(fname, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            spl = line.split(delimiter)
            spl = [s.strip() for s in spl]
            if includes_window:
                tup = (spl[0], spl[1], int(spl[2]))
            else:
                tup = (spl[0], spl[1])
            pairs.append(tup)
    return pairs 


def read_sequences(fname, to_dict=False):
    return parse_fasta(fname, to_dict)


# assumes an h5py file of the embeddings
# returns dictionary of the embeddings to numpy arrays
# note that in this case we assume each (substrate, phospho_site) is annotated by record keys 
def read_embeddings(fname, dtype=torch.float32, device='cpu'):
    embeddings = {}
    with h5py.File(fname, 'r') as f:
        for key in f.keys():
            embedd = torch.from_numpy(f[key][:])
            embedd = embedd.to(dtype)
            embedd = embedd.to(device)
            embeddings[key] = embedd
        
    return embeddings 


# FLAG 274: remove testing parameter for final version
@warn(274)
def build_model_input_from_sequences(ffasta, fpairs, delimiter='\t', dtype=torch.float32, device='cpu', testing=True):
    sequences = read_sequences(ffasta, to_dict=True)
    pairs = read_pairs(fpairs, delimiter, includes_window=True)
    return get_model_input_from_sequences(sequences, pairs, dtype=dtype, device=device, testing=True)


# FLAG 274: remove testing parameter for final version
@warn(274)
def convert_data_seqs_to_embeddings(sequences, pairs, device, model_name, testing=True):
    # get (substrate, site) from pairs 
    subs_to_site = defaultdict(set)
    for _, subseq_id, site in pairs:
        subs_to_site[subseq_id].add(site)

    # start the list off with the full kinase sequences 
    kinase_ids = set([t[0] for t in pairs])
    # note that some in some cases (e.g., panel), kinase embeddings may already be provided
    seqlist = [(k, sequences[k]) for k in kinase_ids if k in sequences]
    
    # now extract the corresponding 15-mers 
    for subseq_id, siteset in subs_to_site.items():
        for site in siteset:
            kmer_id = make_kmer_id(subseq_id, site) 
            subseq = sequences[subseq_id]
            window = extract_window(subseq, site, WINDOW_SIZE, PADDING_CHAR)
            seqlist.append((kmer_id, window))

    # update pairs
    pairs = list(map(lambda t: (t[0], make_kmer_id(t[1], t[2])), pairs))

    # extract the embeddings for all sequences 
    # seq_embeddings = {k: v for k, v in get_embeddings(seqlist, device)}
    seq_embeddings = extract_embeddings(seqlist, device, model_name)

    return pairs, seq_embeddings


# FLAG 274: remove testing parameter for final version
# we probably won't use this function
@warn(274)
def get_model_input_from_sequences(sequences, pairs, dtype=torch.float32, device='cpu', testing=True):
    pairs, seq_embeddings = convert_data_seqs_to_embeddings(sequences, pairs, dtype=torch.float32, device='cpu', testing=True)
    
    # now we can simply return the pair of output embeddings 
    return get_model_input_from_embeddings(seq_embeddings, pairs, dtype=dtype, device=device)


def build_model_input_from_embeddings(fembed, fpairs, delimiter='\t', dtype=torch.float32, device='cpu'):
    # read model embeddings
    seq_embeddings = read_embeddings(fembed)
    # assume each pair embedding is for the appropriate substrate window
    pairs = read_pairs(fpairs, delimiter, includes_window=False)
    return get_model_input_from_embeddings(seq_embeddings, pairs, dtype=torch.float32, device='cpu')


def get_model_input_from_embeddings(seq_embeddings, pairs, dtype=torch.float32, device='cpu'):
    # make the tensors 
    out = [None for _ in range(len(pairs))]
    for i, (kinase_id, substrate_id) in enumerate(pairs):
        x1 = seq_embeddings[kinase_id]
        x2 = seq_embeddings[substrate_id]

        # FLAG 274: substrate goes into first projector
        x = torch.concat([x2, x1])

        # unsqueeze to allow observations concatenated together 
        out[i] = x.unsqueeze(0)

    # concatenate all observations into single 2-D tensor and return 
    return torch.concat(out).to(device)

    
def make_kmer_id(seqid, site):
    return f'{seqid}:psite={site}'
    
