#!/bin/bash

wget http://www.l3s.de/~tempelmeier/geovectors/wdw.pickle.gz
wget http://www.l3s.de/~tempelmeier/geovectors/geovectors_nle.sql.gz
gzip -d wdw.pickle.gz
gzip -d geovectors_nle.sql.gz
