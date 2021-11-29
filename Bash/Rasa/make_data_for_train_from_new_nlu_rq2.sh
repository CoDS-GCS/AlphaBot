while getopts p:f: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        f) file=${OPTARG};;
    esac
done
mkdir $path/re-train/
cp $path/engineer/$file $path/re-train/nlu.yml
cp /vagrant/Dataset/Paper/rules.yml $path/re-train/rules.yml
cp /vagrant/Dataset/Paper/stories.yml $path/re-train/stories.yml
