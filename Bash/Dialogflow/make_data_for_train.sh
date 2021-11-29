while getopts t: flag
do
    case "${flag}" in
        t) timestamp=${OPTARG};;
    esac
done
mkdir /vagrant/Output/$timestamp/train/
cp /vagrant/Output/$timestamp/split/training_data.yml /vagrant/Output/$timestamp/train/nlu.yml
cp /vagrant/Output/$timestamp/data/rules.yml /vagrant/Output/$timestamp/train/rules.yml
cp /vagrant/Output/$timestamp/data/stories.yml /vagrant/Output/$timestamp/train/stories.yml
