while getopts t:o:m: flag
do
    case "${flag}" in
        t) test=${OPTARG};;
        o) out=${OPTARG};;
        m) model=${OPTARG};;
    esac
done
source ~/venv-3.7.9-rasa/bin/activate
cd /vagrant/Rasa-Project/
rasa test nlu --nlu $test --out $out --model $model --config /vagrant/Rasa-Project/config.yml
deactivate
