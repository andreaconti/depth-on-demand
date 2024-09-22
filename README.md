<h1 align="center">
<a href="https://arxiv.org/pdf/2409.08277">Depth on Demand: Streaming Dense Depth from a Low Frame-Rate Active Sensor</a>
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
    <a href="https://arxiv.org/pdf/2409.08277">[Arxiv]</a>
    <a href="https://andreaconti.github.io/projects/depth_on_demand">[Project Page]</a>
</div>
</p>

![](https://andreaconti.github.io/projects/depth_on_demand/images/setup_example.png)

We propose Depth on Demand (DoD), a framework addressing the three major issues related to active depth sensors in streaming dense depth maps: spatial sparsity, limited frame rate and energy consumption of the depth sensors. DoD allows streaming high-resolution depth from an RGB camera and a depth sensor without requiring the depth sensor neither to be dense nor to match the frame rate of the RGB camera. Depth on Demand aims to improve the temporal resolution of an active depth sensor by utilizing the higher frame rate of an RGB camera. It estimates depth for each RGB frame, even for those that do not have direct depth sensor measurements.


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

We provide a GUI (based on [viser](https://viser.studio/latest/#)) to run Depth on Demand on ScannetV2 and 7Scenes sequences interactively. To start the visualizer use the following command and open the browser on the `http://127.0.0.1:8080` as described by the script output.

```bash
$ python visualize/run.py
```

<img width="1275" alt="visualizer_example" src="https://github.com/user-attachments/assets/97eabd71-b6dd-4cb8-b4c2-7029b9b4cb65">

On the right the interface parameters can be configured:

- **Dataset**: select the dataset between scannetv2 and sevenscenes in the drop down menu, the dataset root should be upgraded accordingly.
- **Build Scene**: choose from the drop down menu the scene to use. The other parameters control the density of the input sparse depth both temporally and spatially.
- **TSDF Parameters**: you shouldn't need to change these parameters. They control the mesh reconstruction, tricking the TSDF parameters change the quality of the mesh.

Press **Build Scene** to start the reconstruction. You'll see the system start the reconstruction, you can move the view angle using dragging with the mouse.

**Attention: the interface in this phase may flicker and this step may potentially trigger seizures for people with photosensitive epilepsy. Viewer discretion is advised**

Once the reconstruction is done, playback can be executed multiple times at different frame rates using the **Demo** box and the **Run Demo** button. With the `keep background mesh` option you can either interactively observe the mesh integration growing in time or fix the final one while the point of view moves.

Finally, the **Render Video** button allows to save a demo video and also to **save all the predictions of Depth on Demand** using the `save frames` checkbox. Results are saved in the `renders` directory

## Citation

```bibtex
@InProceedings{DoD_Conti24,
    author    = {Conti, Andrea and Poggi, Matteo and Cambareri, Valerio and Mattoccia, Stefano},
    title     = {Depth on Demand: Streaming Dense Depth from a Low Frame-Rate Active Sensor},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {October},
    year      = {2024},
}
```
