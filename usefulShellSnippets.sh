for i in $(ls -rt /data/atlas/baines/20250310v1_AppliedRecoNN_WithRecoMasses_30_MetCut_RemovedUncertainTruth_WithTagInfo_KeepAllOldSelIncludingNegative/*shape); do echo $i; cat $i; echo ""; echo ""; done


for i in $(ls -rt /data/atlas/baines/*/*410470.memmap.shape); do echo $i; cat $i; echo ""; echo ""; done
for i in $(ls -rt /data/atlas/baines/*/*510115.memmap.shape); do echo $i; cat $i; echo ""; echo ""; done
for i in $(ls -rt /data/atlas/baines/*/*510120.memmap.shape); do echo $i; cat $i; echo ""; echo ""; done