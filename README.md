# GeoVectors Framework
The GeoVectors corpus is a comprehensive large-scale linked open corpus of [OpenStreetMap](ttps://www.openstreetmap.org/) entity embeddings that provides latent representations of over 980 million entities.

This repository contains the GeoVectors Framework that can be used to encode individual OpenStreetMap snapshots. For further information please visit [http://geoemb.l3s.uni-hannover.de/](http://geoemb.l3s.uni-hannover.de/)

# Prerequisites

* Python  >= 3.8 

* [Postgres DB](https://www.postgresql.org/) with [PostGIS](https://postgis.net/)

# Setup

1. Clone the repository
````
git clone https://github.com/NicolasTe/GeoVectors.git
cd GeoVectors
````

2. Install the Python Requirements
```python
pip install -r requirements.txt
```

3. Download the pretrained model
```
cd models/fasttext
./download.sh

```

4. 
Adjust the database credentials in the db.ini file.

# Usage

```
Encoder.py [-h] [--db_cred DB_CRED] [--model MODEL] [--njobs NJOBS] input output modelPath
positional arguments:
  input              Input: directory containing OpenStreetMap snapshots to be encoded
  output             Directory to save encoded files
  modelPath          Path to pretrained model

optional arguments:
  -h, --help         show this help message and exit
  --db_cred DB_CRED  Credentials for the database
  --model MODEL      Name of the embedding model. Either fasttext or nle
  --njobs NJOBS      Number of threads to use

```



1. Download the specific OpenStreetMap snapshot you want to encode, e.g., from [https://download.geofabrik.de/](https://download.geofabrik.de/). We recommend using the osm.pbf format.

2. Run the tag embedding model:
 ```
python3 Encoder.py ./snapshots ./embeddings_tags ./models/fasttext --model fasttext
```
 
 3. Run the NLE embedding model:
 ```
 python3 Encoder.py ./snapshots ./embeddings_nle ./models/nle --model nle --db_cred db.ini
 ```


# License
The MIT License

Copyright 2020 Nicolas Tempelmeier

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
