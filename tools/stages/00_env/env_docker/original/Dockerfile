# FourFusion FLAG 2027: the software environment of every Kaggle run of the submitted system U' = F1s' + RFS.
# BASE = the Kaggle image of those runs (kernel metadata `docker_image`, 2026-11-05; Kaggle tag v173):
#   GPU  gcr.io/kaggle-gpu-images/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01
#   CPU  gcr.io/kaggle-images/python@sha256:0717e81990be591f8c5f98f033d29ab23e9ffde057e5b3a6bd4371f93969e962
# Build from the package root:  docker build -f docker/Dockerfile -t fourfusion:gpu .
#                               docker build -f docker/Dockerfile -t fourfusion:cpu --build-arg FLAVOR=cpu --build-arg BASE=<CPU> .
ARG BASE=gcr.io/kaggle-gpu-images/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01
FROM ${BASE}
ARG FLAVOR=gpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

# 1) What the notebooks' first cells install with `pip install -U` (FLAG_50 pip cell), at the newest release before 2026-11-05.
RUN pip install -U "speechbrain==1.1.2" "onnx==1.23.2" "iopath==0.1.10" "ftfy==6.3.1" "omegaconf==2.4.0" \
        "fvcore==0.1.5.post20221221" "einops==0.8.2" \
 && if [ "$FLAVOR" = gpu ]; then pip install -U "onnxruntime-gpu==1.31.1"; else pip install -U "onnxruntime==1.31.1"; fi

# 2) What they install without -U: kept at the base image's version when the base has it (pip leaves a satisfied package alone).
RUN pip install "regex" "pyyaml" "kaldiio==2.18.1" "silero-vad==6.2.3" "av==19.0.2" "papermill==2.7.0"

# 3) --no-deps Git / pinned installs, at the commits the notebooks use (ImageBind: main on 2026-11-05).
RUN pip install --no-deps "insightface==2.0" \
        "git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c" \
        "git+https://github.com/facebookresearch/ImageBind.git@53680b02d7e37b19b124fa37bae4b6c98c38f5be"

# 4) openai CLIP: `pip install git+…CLIP@d50d76d` fails on this image (setup.py imports pkg_resources; the Kaggle runs log
#    the same error), so the notebooks fall back to a clone in /tmp/openai_clip with that import patched. Done here the same
#    way, so a container needs no network for it (the notebooks skip the clone when /tmp/openai_clip/clip exists).
RUN git clone -q https://github.com/openai/CLIP.git /tmp/openai_clip \
 && git -C /tmp/openai_clip checkout -q d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 \
 && sed -i 's/from pkg_resources import packaging/import packaging, packaging.version/' /tmp/openai_clip/clip/clip.py

# 5) The package code (notebooks, builders, scripts) and the runner; Kaggle's directory layout.
COPY . /opt/fourfusion
RUN mkdir -p /kaggle/input /kaggle/working /kaggle/src \
 && pip freeze > /opt/fourfusion/docker/requirements.lock.txt \
 && python -c "import sys, importlib.util as u; sys.path.insert(0, '/tmp/openai_clip'); import papermill, speechbrain, clip, insightface; \
assert u.find_spec('imagebind') and u.find_spec('wespeaker'); \
print('papermill', papermill.__version__, '| speechbrain', speechbrain.__version__, '| clip', clip.__file__, '| imagebind + wespeaker installed')"
# (imagebind / wespeaker are not imported here: like on Kaggle, the notebooks stub pytorchvideo / s3prl before importing them)
WORKDIR /kaggle/working
CMD ["python", "/opt/fourfusion/docker/ff.py", "plan"]
