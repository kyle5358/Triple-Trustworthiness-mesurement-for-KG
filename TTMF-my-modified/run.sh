# run_python.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu=2
CUDA_VISIBLE_DEVICES=$gpu python Model1.py 
