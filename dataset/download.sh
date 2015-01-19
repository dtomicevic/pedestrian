#!/bin/bash
dbname="PennFudanPed"

if [ -f $dbname.zip ]; then
    rm -f $dbname.zip
fi

if [ -d $dbname ]; then
    rm -rf $dbname
fi

wget http://www.cis.upenn.edu/~jshi/ped_html/$dbname.zip
tar -xzf $dbname.zip
rm $dbname.zip

echo 'the dataset is downloaded and ready to use!'
