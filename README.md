# MuFIC

## Initial Embedding and Side Information
* Initial embedding: please download the crawl-300d-2M.vec.zip from <https://fasttext.cc/docs/en/english-vectors.html>
* Side information: please download from <https://github.com/Yang233666/cmvc>

## Data Preparation
* Cleate a `data` fold
* Download the datasets following <https://github.com/Yang233666/cmvc>, place it under `data`. For ReVerb45K and OPIEC59K datasets, please download from this webpage: <https://drive.google.com/file/d/1vetosoVBf89-It1cD671AfiP0ZrcBz-P/view?usp=sharing>. For NYTimes2018 dataset, please download from this webpage: <https://heathersherry.github.io/ICDE2019_data.html>
* Prepare data: `python preprocessing.py`

## Run
#### Run the main code, you can change datasets and parameters on options.py:
* `python main.py`
