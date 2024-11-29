INSTALLATION INSTRUCTIONS FOR PyMC
----------------------------------

These instructions are kept outside notebooks, because folks may be
unable to read the notebook before installing the software.

This is tested on Ubuntu Linux but should work on all major operating
systems with very minor modifications.

1. Install miniconda:
   https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
   (full anaconda is likely fine too, just AW dislikes bloatware). I used the default location

2. Open a new terminal (or restart shell) and do
   ```
   conda config --set auto_activate_base false
   ```
   to deactivate conda as default environment for Python (otherwise
   all your tools that depend on Python will stop working).

3. Restart terminal to see that the deactivation worked properly

4. Upgrade conda to newest version: `conda update -n base -c defaults conda`

5. Create the prpro container: `conda env create -f environment.yml` (this is slow)

6. Activate it: `conda activate prpro-2025`

7. Reigster the new enviornment as a valid notebook kernel:

```sh
   python -m ipykernel install --user --name prpro-2025 --display-name "prpro-2025"
```

8. To open a notebook from the course go to `bayes/`  directory and
try the following command:

```sh
jupyter lab ./bayes-exercises.ipynb
```

9. In jupyter notebook choose `prpro-2025` as your kernel
