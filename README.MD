# notebook for LAD experiments
## quick start
pull the data from s3 bucket: `s3://aiops-demo-logs/pca-experiments`, fasttext model in the `fasttext_model` folder and k3s data in `input`

``` # install the libs
pip install -r requirements.txt
```

```
# the main notebook
ipython notebook main.ipynb
# notebook for time-window based
ipython notebook time-window.ipynb
```
