# Japanese BART Pretrained Model
We pretrained [BART](https://arxiv.org/pdf/1910.13461.pdf) models with Japanese Wikipedia's text.

## Download 
```shell
# BART_base
wget  
```
They includes a pretrained bart model (`bart_model.pt`), a sentencepiece model (`sp.model`) and a dictionary (`dict.txt`).  

## Requirements for Preprocess
zenhan  
pyknp  
sentencepiece  

## Preprocess
We applied Juman++ to datasets for word segmentation first, and then applied SentencePiece to them for subword segmentation.  
Besides, before applying Juman++, we converted half-width characters to full-width characters.  
  
Prepare datasets (`$TRAIN_SRC`, ...), which contain one sentence per one line.  
```shell
cat $TRAIN_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/train.src-tgt.src
cat $TRIN_TGT | python3 ./jaBART_preprocess.py  --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/train.src-tgt.tgt
cat $VALID_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/valid.src-tgt.src
cat $VALID_TGT | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/valid.src-tgt.tgt
cat $TEST_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/test.src-tgt.src
cat $TEST_TGT | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/test.src-tgt.tgt
cp $DICT $DATASET_DIR/dict.src.txt
cp $DICT $DATASET_DIR/dict.tgt.txt
```

## Finetune
Set `$BART` for `bart_base` or `bart_large`.  
```shell
fairseq-train $DATASET_DIR --arch $BART --restore-file $PRETRAINED_MODEL --save-dir $SAVEPT_DIR \
--task translation_from_pretrained_bart --source-lang src --target-lang tgt \
--criterion label_smoothed_cross_entropy --label-smoothing 0.2 --dataset-impl raw \
--optimizer adam --adam-eps 1e-06 --adam-betas '{0.9, 0.98}' --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
--warmup-updates 2500 --total-num-update 40000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
--max-tokens 1024 --update-freq 2 --save-interval -1 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler  --save-interval-updates 5000 \
--ddp-backend no_c10d --max-update 80000 --patience 3 \
--encoder-normalize-before --decoder-normalize-before
```

## Text Generation
```shell
fairseq-generate $DATASET_DIR --path $FINETUNED_MODEL --task translation_from_pretrained_bart \
--dataset-impl raw --gen-subset test -s src -t tgt --max-sentences 64 > $RESULT
cat $RESULT | grep -P "^H" | cut -f 3- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > ${RESULT}.pred
cat $RESULT | grep -P "^S" | cut -f 2- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > ${RESULT}.src
cat $RESULT | grep -P "^T" | cut -f 2- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > ${RESULT}.tgt
```

## Pretrain
Following command is an example of pretraining bart by yourself.  
`$DATASET_DIR` must contain `train`, `valid` and `dict.txt`.  

```shell
fairseq-train $DATASET_DIR --arch $BART --task denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences 1.0 --replace-length 1 \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
    --max-update 500000 --tensorboard-logdir $TENSORBOARD --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 1 --save-dir $SAVEPT \
    --fp16 --num-workers 0 --fp16-init-scale 4 
```

Following command is an example of pretraining multilingual (ja and en) bart by yourself.  
`$DATASET_DIR` must contain `dict.txt`.  
`$DATASET_DIR/{ja,en}` must contain `train` and `valid`.  
```shell
fairseq-train $DATASET_DIR --arch $BART --task multilingual_denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences 1.0 --replace-length 1 \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
    --max-update 500000 --tensorboard-logdir $TENSORBOARD --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 1 --save-dir $SAVEPT \
    --fp16 --num-workers 0 --fp16-init-scale 4 \
    --langs ja,en --add-lang-token
```

Most of the parameters are according to the papers of [BART](https://arxiv.org/pdf/1910.13461.pdf), [mBART](https://arxiv.org/abs/2001.08210) and [RoBERTa](https://arxiv.org/abs/1907.11692).  
