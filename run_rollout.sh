#!/bin/bash -x

export JAVA_HOME="$HOME/jdk8u222-b10"
export PATH="$JAVA_HOME/bin:$PATH"

PYTHON="${HOME}/MineRL/venv/bin/python"
TRAIN_DIR="${HOME}/rainbow_1/train"
TEST_DIR="${HOME}/rainbow_1/validation"
OUT="${HOME}/minerl_rollouts"

dirs=(
    "${OUT}/train/ep_395"
    "${OUT}/test/val1"
    "${OUT}/train/ep_795"
    "${OUT}/test/val3"
    "${OUT}/train/ep_1295"
    "${OUT}/test/val5"
    "${OUT}/train/ep_1795"
    "${OUT}/test/val7"
    "${OUT}/train/ep_1955"
)

make_video () {
  cd "$1" || exit 1
  ffmpeg -framerate 5 -pattern_type glob -i '*.png' -vcodec libx264 -g 1 output.mp4 && rm ./*.png
}

# 20%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_395_forest420/"
mv saliency "${OUT}/train/ep_395/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val1_mean1.0/"
mv saliency "${OUT}/test/val1/"

# 40%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_795_forest420/"
mv saliency "${OUT}/train/ep_795/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val3_mean0.0/"
mv saliency "${OUT}/test/val3/"

# 60%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_1295_forest420/"
mv saliency "${OUT}/train/ep_1295/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val5_mean0.0/"
mv saliency "${OUT}/test/val5/"

# 80%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_1795_forest420/"
mv saliency "${OUT}/train/ep_1795/"
xvfb-run "$PYTHON" rollout.py --test  --load "${TEST_DIR}/val7_mean0.0/"
mv saliency "${OUT}/test/val7/"

# 100%
xvfb-run "$PYTHON" rollout.py  --load "${TRAIN_DIR}/ep_1955_forest420/"
mv saliency "${OUT}/train/ep_1955/"

for i in "${dirs[@]}"; do
    make_video "$i" && \
    make_video "$i/advantage" && \
    make_video "$i/state" && \
    make_video "$i/total_advantage"
done
