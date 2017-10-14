#!/bin/bash
# wujian@17.10.10

. ./path.sh || exit 1

nj=10
cmd=run.pl
acwt=0.10
beam=13.0
lattice_beam=8.0
min_active=200
max_active=7000 
max_mem=50000000

scoring_opts="--min-lmwt 4 --max-lmwt 15"
model_type="dnn"
left_context=""
right_context=""

. parse_options.sh || exit 1

[ $# -ne 5 ] && echo "format error: $0 <data-dir> <torch-model.pkl> <gmm-model.mdl> <graph-dir> <decode-dir>" && exit 1

data_dir=$1
mdl=$2
gmm_mdl=$3
graph_dir=$4
decode_dir=$5

utils/split_data.sh $data_dir $nj || exit 1
split_data_dir=$data_dir/split$nj

for x in $mdl $gmm_mdl $graph_dir/HCLG.fst $graph_dir/words.txt; do [ ! -f $x ] && echo "error: $x not existed!" && exit 1; done

read_feats="apply-cmvn --norm-vars=true --utt2spk=ark:$split_data_dir/JOB/utt2spk \
    scp:$split_data_dir/JOB/cmvn.scp scp:$split_data_dir/JOB/feats.scp ark:-"

context_opts=""
feats_dim=$(feat-to-dim scp:$split_data_dir/1/feats.scp -) || exit 1

[ ! -z $left_context ] && [ ! -z $right_context ] && read_feats="$read_feats | splice-feats \
    --left-context=$left_context --right-context=$right_context ark:- ark:-" && \
    context_opts="--left-context $left_context --right-context $right_context"

$cmd JOB=1:$nj $decode_dir/log/decode.JOB.log $read_feats \| pytorch/compute-posterior.py \
    --feats-dim $feats_dim --model-type $model_type $context_opts $mdl - - \| \
    latgen-faster-mapped --min-active=$min_active --max-active=$max_active \
    --max-mem=$max_mem --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --allow-partial=true --word-symbol-table=$graph_dir/words.txt \
    $gmm_mdl $graph_dir/HCLG.fst ark:- "ark:|gzip -c > $decode_dir/lat.JOB.gz" || exit 1;

pytorch/score.sh $scoring_opts $data_dir $graph_dir $decode_dir || exit 1;
cp $mdl $decode_dir || exit 1

echo "Decode and scoring done!"
