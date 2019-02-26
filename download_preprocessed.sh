mkdir data

wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz 
#wget https://s3.amazonaws.com/code2vec/data/java-small_data.tar.gz
#wget https://s3.amazonaws.com/code2vec/data/java-med_data.tar.gz
#wget https://s3.amazonaws.com/code2vec/data/java-large_data.tar.gz

tar -xvzf java14m_data.tar.gz
#tar -xvzf java-small_data.tar.gz
#tar -xvzf java-med_data.tar.gz
#tar -xvzf java14m_data.tar.gz

mv java14m data
#mv java-small data
#mv java-med data
#mv java-large data
