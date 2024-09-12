unset http_proxy 
unset https_proxy 
nohup python3 nodectl.py -c 3pc.json up >log.txt 2>&1 & 
export PYTHONPATH=./ && python scripts/flax_sample_hack.py 2>&result.txt &
