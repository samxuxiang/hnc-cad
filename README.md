# Hierarchical Neural Coding for Controllable CAD Model Generation (ICML 2023)

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/)
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://hnc-cad.github.io/) 
[![Youtube](https://img.shields.io/badge/📽️-Video%20-orchid.svg)](https://www.youtube.com/)

*[Xiang Xu](https://samxuxiang.github.io/), [Pradeep Kumar Jayaraman](https://www.research.autodesk.com/people/pradeep-kumar-jayaraman/), [Joseph G. Lambourne](https://www.research.autodesk.com/people/joseph-george-lambourne/), [Karl D.D. Willis](https://www.karlddwillis.com/), [Yasutaka Furukawa](https://www.cs.sfu.ca/~furukawa/)*

![alt HNCode](resources/teaser.png)

> We present a novel generative model for
Computer Aided Design (CAD) that 1) represents high-level design concepts of a CAD model as a
three-level hierarchical tree of neural codes, from global part arrangement down to local curve geometry; and 2) controls the generation of CAD models by specifying the target design using a code tree. Our method supports diverse and higher-quality generation; novel user controls while specifying design intent; and autocompleting a partial CAD model under construction.

<!-- <p align="center">
<img src="https://github.com/threedle/GeoCode/releases/download/v.1.0.0/demo_video_chair.gif" width=250 alt="3D shape recovery"/>
<img src="https://github.com/threedle/GeoCode/releases/download/v.1.0.0/demo_video_vase.gif" width=250 alt="3D shape recovery"/>
<img src="https://github.com/threedle/GeoCode/releases/download/v.1.0.0/demo_video_table.gif" width=250 alt="3D shape recovery"/>
</p>
<p align="center">
A demo video of our program is available on our <a href="https://threedle.github.io/GeoCode/">project page</a>.
</p> -->

## Requirements

### Environment
- Linux
- Python 3.8
- CUDA >= 11.4
- GPU with 24 GB ram recommended

### Dependencies
- PyTorch >= 1.10
- Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core).
- Install other dependencies with ```pip install -r requirements.txt```

We also provide the [docker image](https://hub.docker.com/r/samxuxiang/skexgen). Note: only tested on CUDA 11.4. 


## Dataset 
We use the dataset from [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) for training and evaluation.

The sketch-and-extrude sequences need to be converted to our obj format following the steps from [SkexGen](https://github.com/samxuxiang/SkexGen). 

You can run the following script to download our post-processed raw data 

    python scripts/download.py


After the raw data is downloaded, run this script to get the solid, profile, loop and full CAD Model data

    sh scripts/process.sh


Run the deduplication script, this will output post-filtered data as ```train_deduplicate.pkl```

    sh scripts/deduplicate.sh



## Usage

### Codebook 
Train the three-level codebook with

    sh scripts/codebook.sh

After the codebooks are learned, extract the neural codes corresponding to each training data with

    sh scripts/extract_code.sh

Pretrained weights for the three codebook networks are available [here](https://drive.google.com/file/d/1AA3OLKFgvmmSojyNLzXANw-FPnZLi8x8/view?usp=sharing). You can also download the extracted codes from [here](https://drive.google.com/file/d/1odP_K7l7TilarYgFHFOOIFMGlvlceuc0/view?usp=sharing).


### Random Generation
Run the following script to train the code-tree generator and model generator for unconditional generation

    sh scripts/gen_uncond.sh

For testing, run this script to generate 1000 CAD samples and visualize the results

    sh scripts/sample_uncond.sh

For evaluation, uncomment the eval section in ```sample_uncond.sh```, this would generate > 10,000 samples. Then compute JSD, MMD, and COV scores. Warning: this step can be very slow.

    sh scripts/eval.sh

Please also download the test set from [here](https://drive.google.com/file/d/1FhONYaJTK2vkayfDKH5TaHXDyjl2f4f-/view?usp=sharing) and unzip it inside the ```data``` folder. This is required for computing the evaluation metrics.



### Conditional Generation

Train the full model including model encoder for conditional CAD generation (e.g. autocompletion) 

    sh scripts/gen_cond.sh

For testing, run this script to generate auto-complete resuls from partial sketch and extrude input.

    sh scripts/sample_uncond.sh




## Acknowledgement
This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and DND/NSERC Discovery Grant Supplement, NSERC Alliance Grants, and John R. Evans Leaders Fund (JELF).


## Citation
If you find our work useful in your research, please cite the following paper
```
@inproceedings{xu2023hnc,
  author    = {Xiang Xu, Pradeep Kumar Jayaraman, Joseph G. Lambourne, Karl D.D. Willis, Yasutaka Furukawa},
  title     = {Hierarchical Neural Coding for Controllable CAD Model Generation},
  journal   = {ICML},
  year      = {2023},
}
```

## :hourglass_flowing_sand: UPDATES
- [x] Data preprocess code is released
- [x] Core model code is released
- [x] CAD generation code is released
- [ ] CAD autocompletion code
- [ ] Pretrained weights