# Hierarchical Neural Coding for Controllable CAD Model Generation (ICML 2023)

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/2307.00149)
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://hnc-cad.github.io) 
[![Youtube](https://img.shields.io/badge/📽️-Video%20-orchid.svg)](https://www.youtube.com/watch?v=1XVUJIKioO4)

*[Xiang Xu](https://samxuxiang.github.io/), [Pradeep Kumar Jayaraman](https://www.research.autodesk.com/people/pradeep-kumar-jayaraman/), [Joseph G. Lambourne](https://www.research.autodesk.com/people/joseph-george-lambourne/), [Karl D.D. Willis](https://www.karlddwillis.com/), [Yasutaka Furukawa](https://www.cs.sfu.ca/~furukawa/)*

![alt HNCode](resources/teaser.png)

> We present a novel generative model for
Computer Aided Design (CAD) that 1) represents high-level design concepts of a CAD model as a
three-level hierarchical tree of neural codes, from global part arrangement down to local curve geometry; and 2) controls the generation of CAD models by specifying the target design using a code tree. Our method supports diverse and higher-quality generation; novel user controls while specifying design intent; and autocompleting a partial CAD model under construction.


## Requirements

### Environment
- Linux
- Python 3.8
- CUDA >= 11.4
- GPU with 24 GB ram recommended

### Dependencies
- PyTorch >= 1.10
- Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core) (use mamba if conda is too slow).
- Install other dependencies with ```pip install -r requirements.txt```

We also provide the [docker image](https://hub.docker.com/r/samxuxiang/skexgen). Note: only tested on CUDA 11.4. 


## Dataset 
We use the dataset from [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) for training and evaluation.

The sketch-and-extrude sequences need to be first converted to our obj format following the steps from [SkexGen](https://github.com/samxuxiang/SkexGen). 

Run the following script to download our post-processed DeepCAD data in obj format

    python scripts/download.py


After the data is downloaded, run this script to get the solid, profile, loop and CAD model data

    sh scripts/process.sh


Run the deduplication script, this will output post-filtered data as ```train_deduplicate.pkl```

    sh scripts/deduplicate.sh

Download the ready-to-use [post-deduplicate data](https://drive.google.com/file/d/1U4UuhFzs7BenViVD5tqoQzH72jbE_oKi/view?usp=sharing).



## Usage

### Codebook 
Train the three-level codebook with

    sh scripts/codebook.sh

Download our pretrained codebook module from [here](https://drive.google.com/file/d/1UXvF3fsRM1RxxtArxvBu--t0foU_6ZwR/view?usp=sharing). 

After the codebooks are trained, extract the neural codes corresponding to each training data with

    sh scripts/extract_code.sh

Extracted codes from the pretrained model are available [here](https://drive.google.com/file/d/1uoCcwMGFftgouaH4evg0dDKfS3MtgEIR/view?usp=sharing).


### Random Generation
Run the following script to train the code-tree generator and model generator for unconditional generation

    sh scripts/gen_uncond.sh

Download our pretrained unconditional generation module from [here](https://drive.google.com/file/d/142PMsq3i0mXJMnCkcf-o63DBrU0fe6Sj/view?usp=sharing). 

For testing, run this script to generate 1000 CAD samples and visualize the results

    sh scripts/sample_uncond.sh

For evaluation, uncomment the eval script in ```sample_uncond.sh```, this would generate > 10,000 samples. Then compute JSD, MMD, and COV scores using ```eval.sh```. Warning: this step can be very slow.

    sh scripts/eval.sh

Please also download the [test data](https://drive.google.com/file/d/1FhONYaJTK2vkayfDKH5TaHXDyjl2f4f-/view?usp=sharing) and unzip it inside the ```data``` folder. This is required for computing the evaluation metrics.


### Conditional Generation

Train the full model including model encoder for conditional CAD generation

    sh scripts/gen_cond.sh

For testing (e.g CAD autocomplete), run this script to generate full CAD model from partial extruded profiles.

    sh scripts/sample_cond.sh




## Acknowledgement
This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and DND/NSERC Discovery Grant Supplement, NSERC Alliance Grants, and John R. Evans Leaders Fund (JELF).


## Citation
If you find our work useful in your research, please cite the following paper
```
@inproceedings{xu2023hierarchical,
  title={Hierarchical Neural Coding for Controllable CAD Model Generation},
  author={Xu, Xiang and Jayaraman, Pradeep Kumar and Lambourne, Joseph G and Willis, Karl DD and Furukawa, Yasutaka},
  booktitle={International Conference on Machine Learning},
  pages={38443--38461},
  year={2023}
  publisher={PMLR}
}
```

## Misc 
- If you encounter the issue of "No loop matching the specified signature", try downgrading numpy to 1.23.
