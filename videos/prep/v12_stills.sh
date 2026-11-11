#!/bin/sh
# Render review stills for one of V12–V14 from a prebuilt bundle.
# usage: prep/v12_stills.sh <bundle_dir> <VNN> <frac> <seg_id>...   (renders each segment at start + frac*duration)
B=$1; V=$2; F=$3; shift 3
v=$(echo $V | tr 'V' 'v')
for id in "$@"; do
  fr=$(node -e "
const s=require('./src/$v/script.json'), d=require('./public/audio/$v/durations.json');
let t=s.lead; for(const x of s.segments){ if(x.id==='$id'){ console.log(Math.round((t+d[x.id]*$F)*60)); break;} t+=d[x.id]+(x.pauseAfter??0.45);}")
  npx remotion still "$B" $V out/${v}_$id.png --frame=$fr --log=error >/dev/null 2>&1 || echo "FAIL $id"
done
