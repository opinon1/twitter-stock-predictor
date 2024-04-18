
#!pip install opendatasets --upgrade --quiet

import opendatasets as od


datasets = ['https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020']
datasets.append('https://www.kaggle.com/omermetinn/values-of-top-nasdaq-copanies-from-2010-to-2020')
# Using opendatasets let's download
for dataset in datasets:
    od.download(dataset, "datasets/")
