
# general python imports 
import sys 
from functools import reduce

# for parsing input
from io import StringIO

from .utils import get_model_input_from_sequences
from .utils import get_model_input_from_embeddings

from .utils import read_sequences
from .utils import read_pairs 
from .utils import read_embeddings

from .utils import get_embeddings

from .utils import convert_data_seqs_to_embeddings

# for model 
from .models import load_model
import torch 

# for batching pairs 
from .utils.batch_utils import BatchComp

BATCH_SIZE = 10000
BATCH_HANDLER = BatchComp(BATCH_SIZE)

# for the conditional returns
from .utils.conditional_returns import DISCARD_COMPONENTS
LEN_T = 4
DISCARD = DISCARD_COMPONENTS(LEN_T, True)

# for warnings 
from .utils.warnings import warn

# load model for running 
MODEL = None
MODEL_NAME = {'large': 'esm2_t48_15B_UR50D', 
              'small': 'esm2_t33_650M_UR50D'}


# FLAG 274: remove testing parameter for final version
@warn(274)
def run_kolossus(pairs, seqs=None, embeds=None, dtype=torch.float32, device='cpu', testing=False, 
                 return_projections=False, model='large'):
    global DISCARD

    # device handling 
    device = torch.device(device)

    # 2 things to be done if seqs are input: 
    #     1. convert sequence to embeddings
    #     2. convert elements of pairs from triplets to doublets  
    # task 1 and task 2 done simultaneously in function `convert_data_seqs_to_embeddings`
    if seqs is not None:
        assert isinstance(seqs, dict)
        model_name = MODEL_NAME[model]
        pairs, embeds = convert_data_seqs_to_embeddings(seqs, pairs, device, model_name, testing)
    else:
        assert embeds is not None 

    # don't store projections if not desired 
    if not return_projections:
        DISCARD.add_discard_index(-1, -2)
    else:
        DISCARD = DISCARD_COMPONENTS(LEN_T, True)

    # will operate on pairs in batches 
    result = reduce(lambda x, y: x + y, _run_kolossus(pairs, embeds, dtype, device, model))

    # return either just distances or distances and projections
    out = kolossus_output(result, return_projections)
    return out

    

def kolossus(fpairs, fseqs=None, fembeds=None, dtype=torch.float32, device='cpu', 
             return_projections=False, model='large'):
    # device handling 
    device = torch.device(device)

    if fseqs is not None:
        seqs = read_sequences(fseqs, to_dict=True)
        pairs = read_pairs(fpairs, '\t', includes_window=True)
        return run_kolossus(pairs, seqs=seqs, dtype=dtype, device=device, 
                            return_projections=return_projections, model=model)
    else:
        assert fembeds is not None 
        # read model embeddings
        embeds = read_embeddings(fembeds)
        # assume each pair embedding is for the appropriate substrate window
        pairs = read_pairs(fpairs, '\t', includes_window=False)
        return run_kolossus(pairs, embeds=embeds, dtype=dtype, device=device,
                            return_projections=return_projections, model=model)


# where the magic happens
@BATCH_HANDLER
@DISCARD
def _run_kolossus(pairs, embeds, dtype, device, model):
    """
    model is str: 'large' or 'small'
    """
    global MODEL
    if MODEL is None: 
        MODEL = load_model(device, model)

    X = get_model_input_from_embeddings(embeds, pairs, dtype, device)

    MODEL.eval()
    MODEL.to(device)
    with torch.no_grad():
        Y, kinase_embeds, site_embeds = MODEL(X)

    Y = Y.detach().cpu().numpy().tolist()
    kinase_embeds = kinase_embeds.detach().cpu().numpy()
    site_embeds = site_embeds.detach().cpu().numpy()

    # handle edge case: only one kinase-substrate pair
    if len(pairs) == 1: 
        Y = [Y]

    out = tuple(zip(pairs, Y, kinase_embeds, site_embeds))

    return out

def kolossus_output(result, return_projection):
    if return_projection:
        # get distances for pairs 
        distances = dict(map(lambda t: (t[0], t[1]), result))
        embeddings = {}
        for ((kid, sid), _, kembed, sembed) in result:
            embeddings[kid] = kembed
            embeddings[sid] = sembed
        return distances, embeddings
    else:
        return dict(result)


def set_batch_size(n):
    global BATCH_SIZE
    BATCH_SIZE = n
    try:
        BATCH_HANDLER.set_block_size(BATCH_SIZE)
    except AssertionError:
        print(f"Error: invalid block size passed in: {n}.", file=sys.stderr)
        print("Block size must be an integer of size at least 2", file=sys.stderr)
        sys.exit(1)
    
