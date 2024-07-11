<h1 align="center">
<a href="">Depth on Demand: Streaming Dense Depth from a Low Frame-Rate Active Sensor</a>
</h1>

<p>
<div align="center">
    <a href="https://andreaconti.github.io">Andrea Conti</a>
    &middot;
    <a href="https://mattpoggi.github.io">Matteo Poggi</a>
    &middot;
    <a href="">Valerio Cambareri</a>
    &middot;
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html">Stefano Mattoccia</a>
</div>
<div align="center">
    <a href="">[Arxiv]</a>
    <a href="">[Project Page]</a>
</div>
</p>

## Citation

```bibtex
@InProceedings{
    author    = {Conti, Andrea and Poggi, Matteo and Cambareri, Valerio and Mattoccia, Stefano},
    title     = {Depth on Demand: Streaming Dense Depth from a Low Frame-Rate Active Sensor},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {October},
    year      = {2024},
}
```

## Evaluation Code

In this repo we provide __evaluation__ code for our paper, it allows to load the pre-trained models on ScanNetV2, TartanAir and KITTI. Please note that <u>we do not provide the source code of our models</u> but only compiled binaries to perform inference.


### Install

Dependencies can be installed with `conda` or `mamba` as follows:

```bash
$ git clone https://github.com/andreaconti/depth-on-demand.git
$ cd depth-on-demand
$ conda env create -f environment.yml  # use mamba  if conda is too slow
$ conda activate depth-on-demand
# then, download and install the wheel containing the pretrained models, available for linux, windows and macos
$ pip install https://github.com/andreaconti/depth-on-demand/releases/download/models%2Fv0.1.1/depth_on_demand-0.1.1-cp310-cp310-linux_x86_64.whl --no-deps
```

We provide scripts to prepare the datasets for evaluation, each automatically downloads and unpack the dataset in the `data` directory:

```bash
# as an example to download the evaluation sequences of scannetv2 can be used the following
$ python download/scannetv2.py
```

### Evaluate

To evaluate the framework we provide the following script, which loads the specified dataset, its configuration and returns the metrics.

```bash
$ python evaluate/metrics.py --dataset scannetv2
```

Configuration for each dataset is in `evaluate/config`

### Visualize

We provide a GUI to run Depth on Demand on ScannetV2 and 7Scenes sequences interactively. To start the visualizer use the following command and open the browser on the `http://127.0.0.1:8080` as described by the script output.

```bash
$ python visualize/run.py
```