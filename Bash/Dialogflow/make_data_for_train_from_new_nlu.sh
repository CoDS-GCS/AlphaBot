while getopts t:f: flag
do
    case "${flag}" in
        t) timestamp=${OPTARG};;
        f) file=${OPTARG};;
    esac
done
mkdir $timestamp/re-train/
cp $timestamp/engineer/$file $timestamp/re-train/nlu.yml
