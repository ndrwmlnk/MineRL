#!/bin/bash -x

export JAVA_HOME="$HOME/jdk8u222-b10"
export PATH="$JAVA_HOME/bin:$PATH"

PYTHON="${HOME}/MineRL/venv/bin/python"
XRUN=" ${PYTHON}"
TRAIN_DIR="${HOME}/rainbow_1/train"
TEST_DIR="${HOME}/rainbow_1/validation"
OUT="${HOME}/minerl_rollouts"

dirs=(
    "${OUT}/train/ep_199"
    "${OUT}/test/val1"
    "${OUT}/train/ep_399"
    "${OUT}/test/val3"
    "${OUT}/train/ep_599"
    "${OUT}/test/val5"
    "${OUT}/train/ep_799"
    "${OUT}/test/val7"
    "${OUT}/test/val9"
)

make_video () {
  cd "$1" || exit 1
  ffmpeg -framerate 5 -pattern_type glob -i '*.png' -vcodec libx264 -g 1 output.mp4 # && rm *.png
}

# 100%
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val9_mean0.0/"
mv saliency "${OUT}/test/val9/"

# 20%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_199_forest600506/"
mv saliency "${OUT}/train/ep_199/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val1_mean2.0/"
mv saliency "${OUT}/test/val1/"

# 40%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_399_forest600506/"
mv saliency "${OUT}/train/ep_399/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val3_mean3.0/"
mv saliency "${OUT}/test/val3/"

# 60%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_599_forest600506/"
mv saliency "${OUT}/train/ep_599/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val5_mean2.0/"
mv saliency "${OUT}/test/val5/"

# 80%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_799_forest600506/"
mv saliency "${OUT}/train/ep_799/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val7_mean5.0/"
mv saliency "${OUT}/test/val7/"


for i in "${dirs[@]}"; do
    make_video "$i" && \
    make_video "$i/advantage" && \
    make_video "$i/state" && \
    make_video "$i/total_advantage"
done
