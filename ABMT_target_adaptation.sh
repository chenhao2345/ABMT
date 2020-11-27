SOURCE=$1
TARGET=$2
ARCH=$3

if [ $# -ne 3 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ABMT_target_adaptation.py -dt ${TARGET} -ds ${SOURCE} -a ${ARCH} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 --dropout 0 \
	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain/model_best.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-ABMT --features 0
