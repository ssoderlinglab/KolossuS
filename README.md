# KolossuS: Kinase Signaling Prediction Tool

Deconvolving the substrates of hundreds of kinases linked to phosphorylation networks driving
cellular behavior is a fundamental, unresolved biological challenge, largely due to the poorly
understood interplay of kinase selectivity and substrate proximity. We introduce KolossuS, a
deep learning framework leveraging protein language models to decode kinase-substrate
specificity. KolossuS achieves superior prediction accuracy and sensitivity across mammalian
kinomes, enabling proteome-wide predictions and evolutionary insights. By integrating
KolossuS with CRISPR-based proximity proteomics in vivo, we capture kinase-substrate
recognition and spatial context, obviating prior limitations. We show this combined framework
identifies kinase substrates associated with physiological states such as sleep, revealing both
known and novel Sik3 substrates during sleep deprivation. This novel integrated
computational-experimental approach promises to transform systematic investigations of
kinase signaling in health and disease.


## Preprint
**Jha K., Shonai D., Parekh A., Uezu A., Fujiyama T., Yamamoto H., Parameswaran P., Yanagisawa M., Singh R., Soderling S. (2025). Deep Learning-coupled Proximity Proteomics to Deconvolve Kinase Signaling In Vivo. bioRxiv, 2025-04. [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.04.27.650849v1)**


## Installation
```
pip install kolossus
```

## Usage 
Run `KolossuS` on the command line using `kolossus-cli`:

```
usage: kolossus-cli [-h] -p PAIRS [-s SEQS] [-e EMBEDDINGS] [--dtype DTYPE] [-r PROJECTIONS] [-d DEVICE]
                    [--batch_size BATCH_SIZE] -o OUTPUT [--model-small]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS, --pairs PAIRS
                        format: <kinase_id> <substrate_id> <substrate_phosphorylation_site>
  -s SEQS, --seqs SEQS  fasta formatted file of sequences, either seqs or embeddings must be provided
  -e EMBEDDINGS, --embeddings EMBEDDINGS
                        h5 file of sequence embeddings, either seqs or embeddings must be provided
  --dtype DTYPE         data type of sequence embeddings (usually float32)
  -r PROJECTIONS, --projections PROJECTIONS
                        name of .h5 files for kolossus projections
  -d DEVICE, --device DEVICE
                        default device on which to run model
  --batch_size BATCH_SIZE
                        Number of pairs at a time on which to run model
  -o OUTPUT, --output OUTPUT
                        desired file path for output
  --model-small         flag to use the KolossuS model that uses ESM2-650M parameters
```

Note that the fasta file should contain **all** of the sequences (including the full substrate sequences). 
We'll get the appropriate windows from the pairs file. 

To get the ESM-2 embeddings for your protein sequences, you can use the `kolossus-extract` command. 

```
usage: kolossus-extract [-h] -i I [--model MODEL] [--device DEVICE] -o O

optional arguments:
  -h, --help       show this help message and exit
  -i I             name of input fasta file
  --model MODEL    name of the ESM-2 model for which you want embeddings: esm2_t48_15B_UR50D or esm2_t33_650M_UR50D
  --device DEVICE  cpu or gpu device to use
  -o O             name of output .h5 file
```

The `pairs` file should be formatted like so:

```
kinase_1  substrate_1  substrate_1_phosphorylation_site_1
kinase_1  substrate_1  substrate_1_phosphorylation_site_2
kinase_2  substrate_2  substrate_2_phosphorylation_site_1
...
```

Here, `<substrate_x_phosphorylation_site_y>` is the offset of the phosphorylated residue. 
So for example, if the substrate has sequence 'GGRGSDD', and the serine (5th amino acid)
is the phosphorylated residue, then `substrate_phosphorylation_site=5`.


There is also a python interface for using `KolossuS` within kolossus scripts. Main function for usage is `kolossus`. Function works as follows: 

```
Input:
  - fasta file of all sequences (or .h5 file of embeddings)
  - pair file of format '<kinase_id>\t<substrate_id>\t<substrate_phosphorylation_site>'
  - device
  - model: "large" (6B parameter ESM2 base) or "small" (650M parameter ESM2 base)

Output:
  - pairs (kinase_id, substrate_id, substrate_phosphorylation_site, predicted_probability)
```

Usage:

``` python
## on the command line
kinase_file="kinases.fasta"
substrate_file="substrates.fasta"

cat $kinase_file $substrate_file > seqs.fasta

## in python
from kolossus import kolossus

# define inputs to function
seqs_file = 'seqs.fasta' 
pairs_file = 'pairs_with_phosphorylation_sites.txt'

# returns a dictionary (kinase, substrate, site): probability
pairs_and_probs = kolossus(pairs_file, fseqs=seqs_file, device='cpu')

# to get kolossus embeddings: use the return_projections parameter
pairs_and_probs, projections = kolossus(pairs_file, fseqs=seqs_file, device='cpu', return_projections=True)
```
