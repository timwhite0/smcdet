### Probabilistic object detection with sequential Monte Carlo samplers

Let $x$ denote an image. Let $s$ denote the number of objects in the image, and let $z_{(s)} = \\{\ell_{(s)}^j, f_{(s)}^j\\}$ denote the properties of the $s$ objects, where $\ell_j$ is the location of object $j$ and $f_j$ are the other features of object $j$ (e.g, brightness, shape). Let $s \cup z_{(s)}$ denote a catalog of latent random variables that describes the image. Assume that a domain-specific forward model (i.e., a prior $p(s) p(z_{(s)} \vert s)$ and a likelihood $p(x \mid z_{(s)}, s)$ can be evaluated for any particular $x$ and $s \cup z_{(s)}$. We use [sequential Monte Carlo samplers](https://www.tandfonline.com/doi/full/10.1080/01621459.2022.2087659) to sample from the posterior $p(s, z_{(s)} \mid x)$.

Our motivating scientific example is the detection and [deblending](https://www.nature.com/articles/s42254-021-00353-y) of stars in astronomical images. Please see `notebooks/example.ipynb` and `experiments` for some examples. Other potential applications of our algorithm include cell detection in microscopy images and tree crown delineation in satellite images.

To use the code in this repository, please follow these steps:

1. Install [poetry](https://python-poetry.org/docs/)
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone `smcdet`
```
git clone git@github.com:timwhite0/smcdet.git
```

3. Create a virtual environment with all required dependencies
```
cd smcdet
export POETRY_VIRTUALENVS_IN_PROJECT=1
poetry install
```

4. Activate the environment (run `poetry env activate` first to ensure that the following command is correct)
```
source .venv/bin/activate
```
