while getopts p: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
    esac
done
mkdir $path/train/
cp $path/split/training_data.yml $path/train/nlu.yml
cp $path/data/rules.yml $path/train/rules.yml
cp $path/data/stories.yml $path/train/stories.yml
