while getopts d:o: flag
do
    case "${flag}" in
        d) data=${OPTARG};;
        o) out=${OPTARG};;
    esac
done
source ~/venv-3.7.9-rasa/bin/activate
cd /vagrant/Rasa-Project/
rasa train --data $data --out $out --config /vagrant/Rasa-Project/config.yml
deactivate
