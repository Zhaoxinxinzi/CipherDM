FROM secretflow/ubuntu-base-ci:latest

# ENV MYPATH /flax_diffusion
# WORKDIR $MYPATH

RUN pip install flax
RUN pip install jax
RUN pip install imageio
RUN pip install pickle
RUN pip install tqdm

COPY CipherDM/ /code/CipherDM/

RUN python3 nodectl.py -c 3pc.json up
RUN python3 flax_sample_images.py

ENTRYPOINT ["/code/CipherDM/eval.sh"]
