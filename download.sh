mkdir data
wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz 
wget https://s3.amazonaws.com/code2seq/datasets/java-med.tar.gz
wget https://s3.amazonaws.com/code2seq/datasets/java-large.tar.gz
wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz
tar -xvzf java-small.tar.gz
tar -xvzf java-med.tar.gz
tar -xvzf java-large.tar.gz
tar -xvzf java14m_data.tar.gz
mv java-small data
mv java-med data
mv java-large data
rm java-small.tar.gz
rm java-med.tar.gz
rm java-large.tar.gz
rm java14m_data.tar.gz
