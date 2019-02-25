mkdir data

#downloading data
wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz 
wget https://s3.amazonaws.com/code2seq/datasets/java-med.tar.gz
#wget https://s3.amazonaws.com/code2seq/datasets/java-large.tar.gz

#extracting data
tar -xvzf java-small.tar.gz
tar -xvzf java-med.tar.gz
#tar -xvzf java-large.tar.gz

#moving to data folder
mv java-small data
mv java-med data
#mv java-large data

#removing archives
rm java-small.tar.gz
rm java-med.tar.gz
#rm java-large.tar.gz

#delete non .java files
find data/java-small -type f ! -name "*.java" -delete
find data/java-med -type f ! -name "*.java" -delete
#find data/java-large -type f ! -name "*.java" -delete

#delete all empty folders
find data/java-small -type d -empty -delete
find data/java-med -type d -empty -delete
#find data/java-large -type d -empty -delete

#flatten folders
python flatten.py

#delete all empty again
find data/java-small -type d -empty -delete
find data/java-med -type d -empty -delete
#find data-java-large -type d -empty -delete

#balance
python balance.py

#balance again to be sure
python balance.py
