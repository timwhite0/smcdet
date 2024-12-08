### Sequential Monte Carlo for probabilistic object detection

This repository examines the task of detecting scientifically meaningful objects in images.

Let $x$ denote a noisy image. Let $z := \\{s, \\{\ell_j, f_j\\}_{j=1}^s\\}$ denote a catalog of latent random variables that describe the image, where $s$ is the number of objects in the image, $\ell_j$ is the location of object $j$, and $f_j$ are the features of object $j$ (e.g., brightness, shape). Assume that a domain-specific forward model (i.e., a prior $p(z)$ and a likelihood $p(x \mid z))$ can be evaluated for any particular $x$ and $z$. We use [sequential Monte Carlo samplers](https://www.tandfonline.com/doi/full/10.1080/01621459.2022.2087659) to sample from the posterior $p(z \mid x)$.

Our motivating scientific example is the detection and [deblending](https://www.nature.com/articles/s42254-021-00353-y) of stars in crowded astronomical images. Please see `notebooks/example.ipynb` and `experiments` for some examples. Other potential applications of our algorithm include cell detection in microscopy images and tree crown delineation in satellite images.

![](https://i.ibb.co/tYbQK84/Unknown.png)

To ensure that all of the notebooks and scripts in this repository run as intended, please make sure to create a new virtual environment and install all required packages:
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
