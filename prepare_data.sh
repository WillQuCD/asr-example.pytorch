#!/bin/bash
# wujian@17.10.5

[ $# -ne 3 ] && echo "format error: $0: <data-dir> <ali-dir> <dst-dir>" && exit 1

. ./path.sh || exit 1

src_dir=$1
ali_dir=$2
dst_dir=$3
tmp_dir=ali

[ ! -d $dst_dir ] && echo "$0: $dst_dir not exist" && mkdir -p $dst_dir

nj=$(cat $ali_dir/num_jobs) || exit 1


run.pl JOB=1:$nj $tmp_dir/log/align.JOB.log gunzip -c $ali_dir/ali.JOB.gz \| ali-to-pdf \
    $ali_dir/final.mdl ark:- ark,scp:$tmp_dir/ali.JOB.ark,$tmp_dir/ali.JOB.scp || exit 1

cat $tmp_dir/ali.*.scp | sort > $tmp_dir/ali.scp && cat $src_dir/feats.scp | utils/shuffle_list.pl | head -n500 | awk '{print $1}' | sort > $dst_dir/split.id || exit 1

utils/filter_scp.pl --exclude  $dst_dir/split.id $tmp_dir/ali.scp | sort > $dst_dir/train.ali.scp || exit 1
utils/filter_scp.pl $dst_dir/split.id $tmp_dir/ali.scp | sort > $dst_dir/test.ali.scp || exit 1

awk '{print $1}' $dst_dir/test.ali.scp > $dst_dir/test.ali.id & awk '{print $1}' $dst_dir/train.ali.scp > $dst_dir/train.ali.id || exit 1

utils/filter_scp.pl $dst_dir/train.ali.id $src_dir/feats.scp | sort > $dst_dir/train.feats.scp || exit 1
utils/filter_scp.pl $dst_dir/test.ali.id $src_dir/feats.scp | sort > $dst_dir/test.feats.scp || exit 1

rm $dst_dir/*.id || exit 1

for x in train test; do
    copy-int-vector scp:$dst_dir/${x}.ali.scp ark:$dst_dir/${x}_labels.ark || exit 1
    apply-cmvn --norm-vars=true --utt2spk=ark:$src_dir/utt2spk scp:$src_dir/cmvn.scp scp:$dst_dir/${x}.feats.scp \
        ark:$dst_dir/${x}_inputs.ark || exit 1
done 

rm -rf $tmp_dir || exit 1
echo "Done"
