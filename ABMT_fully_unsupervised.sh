TARGET=$1
ARCH=$2

if [ $# -ne 2 ]
  then
    echo "Arguments error: <TARGET> <ARCH>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/ABMT_target_adaptation.py -dt ${TARGET} -a ${ARCH} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 --dropout 0 \
	--logs-dir logs/${TARGET}/${ARCH}-ABMT --features 0 --no-source
