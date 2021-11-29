while getopts m:d:o: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) data=${OPTARG};;
        o) out=${OPTARG};;
    esac
done
source ~/venv-3.7.9-rasa/bin/activate
cd /vagrant/Rasa-Project/
rasa train nlu --finetune $model --config config.yml --nlu $data --out $out --epoch-fraction 0.5
deactivate
