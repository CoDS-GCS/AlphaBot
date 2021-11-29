while getopts t:f: flag
do
    case "${flag}" in
        t) timestamp=${OPTARG};;
        f) file=${OPTARG};;
    esac
done
mkdir /vagrant/Output/$timestamp/re-train/
cp /vagrant/Output/$timestamp/engineer/$file /vagrant/Output/$timestamp/re-train/nlu.yml
cp /vagrant/Output/$timestamp/data/rules.yml /vagrant/Output/$timestamp/re-train/rules.yml
cp /vagrant/Output/$timestamp/data/stories.yml /vagrant/Output/$timestamp/re-train/stories.yml
