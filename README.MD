# notebook for LAD experiments
## quick start
git clone this repo.
get the data from s3 bucket: `s3://aiops-demo-logs/lad-data`.
put fasttext model in `fasttext_model` folder and input k3s data in `input`. 

``` 
# install the libs
pip install -r requirements.txt
```

```
# the main notebook
ipython notebook main.ipynb
# notebook for time-window based
ipython notebook time-window.ipynb
```
