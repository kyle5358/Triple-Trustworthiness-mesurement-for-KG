# run_python.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu=1
CUDA_VISIBLE_DEVICES=$gpu python Model1_cul_new.py 
