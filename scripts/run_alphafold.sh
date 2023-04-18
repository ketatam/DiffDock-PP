#!/bin/bash
# conda activate /data/rsg/nlp/rmwu/miniconda3/envs/af

strings=("/data/rsg/nlp/sdobers/ruslan/fasta_files/a9_1a95.pdb1_3.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/cf_2cfx.pdb1_12.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ed_2ed3.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/kq_1kq1.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/p7_1p7z.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sk_1sk6.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/tn_1tnu.pdb6_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/a9_1a9x.pdb4_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/cf_2cfx.pdb1_7.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ed_2ed5.pdb2_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/kq_1kq1.pdb1_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/p7_2p7v.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sk_1sk6.pdb2_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/tn_3tnp.pdb1_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/a9_2a9w.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/cf_3cf0.pdb1_6.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ed_2ed6.pdb4_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/kq_3kqf.pdb2_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/p7_3p7x.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sk_1sk6.pdb3_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/tn_3tnz.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/a9_3a98.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/cf_3cf0.pdb2_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ed_3edp.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/kq_3kqx.pdb1_10.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/p7_4p7c.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sk_3sk1.pdb2_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/tn_4tn5.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/a9_3a9s.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/cf_5cff.pdb2_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ed_4edz.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/kq_3kqx.pdb1_11.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/p7_4p7s.pdb1_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sk_3sk2.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/tn_4tnp.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/aq_2aq6.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/dm_1dm3.pdb1_4.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/hm_2hmf.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/nq_1nq3.pdb2_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/pj_1pj1.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sl_1slt.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ww_1ww6.pdb2_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/aq_2aqp.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/dm_3dmp.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/hm_2hmf.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/nq_1nqm.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/pj_1pjl.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sl_3sll.pdb1_12.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ww_1wwr.pdb3_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/aq_4aq2.pdb1_10.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/dm_3dmp.pdb3_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/hm_3hm2.pdb2_3.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/nq_1nqu.pdb1_9.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/pj_1pjl.pdb2_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sl_3sll.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ww_2ww2.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/aq_4aqa.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/dm_5dm3.pdb7_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/hm_3hmu.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/nq_2nq2.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/pj_3pj5.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sl_3sll.pdb1_6.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ww_2ww2.pdb1_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/aq_4aqn.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/dm_5dm7.pdb1_22.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/hm_4hm1.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/nq_3nqn.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/pj_4pj1.pdb1_55.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sl_3slp.pdb1_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/ww_4wwa.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/b2_1b26.pdb1_3.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/eb_1ebo.pdb2_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/j3_1j30.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/oi_2oie.pdb1_5.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/s5_1s57.pdb1_10.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sw_1swd.pdb1_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/b2_1b26.pdb1_9.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/eb_2eb6.pdb1_3.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/j3_2j3g.pdb2_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/oi_3oix.pdb3_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/s5_1s57.pdb1_4.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sw_1swl.pdb1_5.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/b2_2b24.pdb1_8.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/eb_4eba.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/j3_2j3g.pdb2_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/oi_4oip.pdb1_3.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/s5_1s57.pdb1_8.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sw_1swp.pdb1_3.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/b2_2b24.pdb1_9.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/eb_4eba.pdb1_2.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/j3_4j38.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/oi_4oip.pdb1_8.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/s5_1s57.pdb2_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sw_3sw5.pdb1_8.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/b2_2b2e.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/eb_4eba.pdb3_1.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/j3_4j3f.pdb5_5.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/oi_4oiz.pdb1_0.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/s5_1s5z.pdb1_5.fasta" "/data/rsg/nlp/sdobers/ruslan/fasta_files/sw_3swc.pdb3_0.fasta")

# check if the script is invoked with an argument
if [ $# -eq 0 ]; then
  echo "Usage: $0 <number>"
  exit 1
fi

# check if the argument is a number between 0 and 9
if ! [[ $1 =~ ^[0-9]$ ]]; then
  echo "Error: the argument must be a number between 0 and 9."
  exit 1
fi

# define the range of indices to select
#start=$((8*$1 + 20))
#end=$((8*$1 + 8 + 20))
start=90
end=100

#gpu=$(($1/2+2))
gpu=1

selected=("${strings[@]:$start:$((end-start+1))}")

# save the selected strings into a comma-separated string
string=$(IFS=','; echo "${selected[*]}")

# print the selected strings as a comma-separated string
echo "Selected strings: $string"
echo "GPU $gpu"

export out_dir=/data/rsg/nlp/sdobers/ruslan/alphafold_results
#export gpu
export num_pred=1  # I recommend 1 for speed

python /data/rsg/nlp/sdobers/ruslan/alphafold/docker/run_docker.py \
  --docker_image_name "alphafold" \
  --fasta_paths=$string \
  --data_dir=/scratch/rmwu/data/alphafold_db \
  --model_preset=multimer \
  --max_template_date=2023-02-01 \
  --output_dir=$out_dir \
  --num_multimer_predictions_per_model=$num_pred \
  --gpu_devices=$gpu
