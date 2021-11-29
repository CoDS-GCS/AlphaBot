#!/usr/bin/env bash

export DEBIAN_FRONTEND=noninteractive
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.8.0
echo '. $HOME/.asdf/asdf.sh' >> ~/.bashrc
echo '. $HOME/.asdf/completions/asdf.bash' >> ~/.bashrc
source ~/.bashrc
set +u
source ~/.asdf/asdf.sh
set -u
asdf plugin-add python
asdf install python 3.7.9
asdf global python 3.7.9
sudo apt-get install -y python3-testresources
pip3 install --no-input --upgrade pip
pip3 install --no-input setuptools
pip3 install --no-input setuptools -U
python3 -m venv ~/venv-3.7.9
source ~/venv-3.7.9/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --no-input -r /vagrant/requirements.txt
python -c "import nltk; nltk.download('popular')"
python -c "import gensim.downloader as api; api.load('glove-twitter-25');"
#### Start adding TextBlob data ####
python -m textblob.download_corpora
#### End adding TextBlob data ####
deactivate
python3 -m venv ~/venv-3.7.9-rasa
echo 'RASA_X_PASSWORD="123456"' >> ~/venv-3.7.9-rasa/bin/activate
echo 'export RASA_X_PASSWORD' >> ~/venv-3.7.9-rasa/bin/activate
source ~/venv-3.7.9-rasa/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --no-input -r /vagrant/Rasa-Project/requirements.txt --extra-index-url https://pypi.rasa.com/simple
cd /vagrant/Rasa-Project/ && rasa telemetry disable
deactivate
#### Start Installing Duckling ####
cd /home/vagrant/ && wget -qO- https://get.haskellstack.org/ | sh ### Installing Stack needed for installing Duckling
cd /home/vagrant/ && git clone https://github.com/facebook/duckling.git ### Cloning Duckling
cd /home/vagrant/duckling/ && stack build ### Installing Duckling using Stack
#### End Installing Duckling ####
# sudo apt-get install -y default-jdk default-jre
git config --global user.name "contributor" # my user: farhour
git config --global user.email "contributor@domain.com" # my email: farbod@farhour.com
echo " ****** We are done !!! ********"
