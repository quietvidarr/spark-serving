#!/bin/bash
# Resumable image pull for flaky links: downloads each blob with curl -C - (resume)
# and retries, then assembles a docker-archive and `docker load`s it.
# usage: resumable-pull.sh <registry> <repo> <tag> <workdir>
#   e.g. resumable-pull.sh ghcr.io <org>/<image> latest ~/recipes/.imgcache/q36
set -euo pipefail
REG=$1 REPO=$2 TAG=$3 DIR=$4
mkdir -p "$DIR/blobs"; cd "$DIR"
ACCEPT='application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'

token() {
  if [ "$REG" = "ghcr.io" ]; then curl -s "https://ghcr.io/token?scope=repository:$REPO:pull" | jq -r .token
  else curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | jq -r .token; fi
}
API=https://$REG/v2/$REPO
[ "$REG" = "docker.io" ] && API=https://registry-1.docker.io/v2/$REPO

T=$(token)
curl -sf -H "Authorization: Bearer $T" -H "Accept: $ACCEPT" "$API/manifests/$TAG" > top.json
if jq -e '.manifests' top.json >/dev/null; then
  D=$(jq -r '.manifests[] | select(.platform.architecture=="arm64" and .platform.os=="linux") | .digest' top.json | head -1)
  curl -sf -H "Authorization: Bearer $T" -H "Accept: $ACCEPT" "$API/manifests/$D" > manifest.json
else cp top.json manifest.json; fi

fetch() {  # digest expected_size
  local dg=$1 size=$2 f="blobs/${1#sha256:}"
  for i in $(seq 1 200); do
    [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$size" ] && break
    [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt "$size" ] && rm -f "$f"
    T=$(token)
    curl -sSL --retry 5 --connect-timeout 20 --speed-limit 10000 --speed-time 60 -C - \
      -H "Authorization: Bearer $T" "$API/blobs/$dg" -o "$f" || { echo "retry $i $dg"; sleep 5; }
  done
  echo "$dg  $f" | sed 's/^sha256://' | sha256sum -c --quiet - || { echo "BAD CHECKSUM $dg"; rm -f "$f"; exit 1; }
}

CFG=$(jq -r .config.digest manifest.json); fetch "$CFG" "$(jq -r .config.size manifest.json)"
n=$(jq '.layers|length' manifest.json); k=0
jq -r '.layers[] | "\(.digest) \(.size)"' manifest.json | while read dg size; do
  k=$((k+1)); fetch "$dg" "$size"; echo "layer $k/$n ok"
done

# docker-archive (docker save format) with compressed layers; docker load handles gzip/zstd layers
mkdir -p archive; rm -f archive/*
ln -f "blobs/${CFG#sha256:}" archive/config.json
LAYERS=$(jq -r '.layers[].digest' manifest.json | while read dg; do ln -f "blobs/${dg#sha256:}" "archive/${dg#sha256:}.tar"; echo "\"${dg#sha256:}.tar\""; done | paste -sd,)
echo "[{\"Config\":\"config.json\",\"RepoTags\":[\"$REG/$REPO:$TAG\"],\"Layers\":[${LAYERS}]}]" > archive/manifest.json
[ "$REG" = "docker.io" ] && sed -i "s#docker.io/##" archive/manifest.json
tar -C archive -cf - . | docker load
echo "RESUMABLE PULL OK $REG/$REPO:$TAG"
