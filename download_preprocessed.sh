mkdir data

wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz 
wget https://s3.amazonaws.com/code2vec/data/java-small_data.tar.gz
wget https://s3.amazonaws.com/code2vec/data/java-med_data.tar.gz
#wget https://s3.amazonaws.com/code2vec/data/java-large_data.tar.gz

tar -xvzf java14m_data.tar.gz
tar -xvzf java-small_data.tar.gz
tar -xvzf java-med_data.tar.gz
#tar -xvzf java14m_data.tar.gz

mv java14m_data data
mv java-small_data data
mv java-med_data data
#mv java-large_data data
