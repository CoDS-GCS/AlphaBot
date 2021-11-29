while getopts f:t:p: flag
do
    case "${flag}" in
        f) fraction=${OPTARG};;
        t) testing=${OPTARG};;
        p) path=${OPTARG};;
    esac
done
# echo "fraction: $fraction";
source ~/venv-3.7.9-rasa/bin/activate
cd /vagrant/Rasa-Project/
# rasa data split nlu --training-fraction $fraction --out $path/split --nlu $path/data --random-seed 7322
rasa data split nlu --training-fraction $fraction --out $path/split --nlu $path/data
mkdir $path/temporary/
cp $path/split/test_data.yml $path/temporary/nlu.yml
cp $path/data/rules.yml $path/temporary/rules.yml
cp $path/data/stories.yml $path/temporary/stories.yml
mkdir $path/testing_validation/
# rasa data split nlu --training-fraction $testing --out $path/testing_validation --nlu $path/temporary --random-seed 7322
rasa data split nlu --training-fraction $testing --out $path/testing_validation --nlu $path/temporary
rm -rf $path/temporary/
deactivate
