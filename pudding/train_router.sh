CUDA_VISIBLE_DEVICES=5 /home/jim/anaconda3/envs/llama3/bin/python train_BERT_likelihood_MSE.py \
 --epochs 10 \
 --csv_files ./all_log.csv \
 --output_dir result/router
