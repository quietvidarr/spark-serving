#!/bin/bash
# Image helper for the TensorFold recipe (used by the model router, see models.json).
# The recipe serves a local tag ($IMAGE) that scripts/prepare.sh pulls from GHCR under a tag derived
# from the recipe's own patches ($GHCR_IMAGE:<TensorFold version>-<patches hash>), so "is there a newer
# image" means "does the registry's image for the current patches differ from the local one".
#   tensorfold-image.sh ref      print "REGISTRY_REF LOCAL_NAME" (models.json image_cmd)
#   tensorfold-image.sh update   pull that ref with retries (Wi-Fi), tag it, let prepare.sh verify it
# Run from the recipe directory.
set -euo pipefail
source ./scripts/config.sh
ref="$GHCR_IMAGE:${TF_VERSION}-$(patches_hash)"
case "${1:-ref}" in
  ref) echo "$ref $IMAGE" ;;
  update)
    for i in $(seq 1 20); do
      docker pull "$ref" && { docker tag "$ref" "$IMAGE"; exec ./scripts/prepare.sh; }
      sleep 20
    done
    echo "could not pull $ref"; exit 1 ;;
  *) echo "usage: $0 ref|update"; exit 2 ;;
esac
