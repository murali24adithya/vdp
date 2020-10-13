# /bin/bash
# Echo "pulling"
# scp -r atharva2@cc-login1.campuscluster.illinois.edu:/home/atharva2/scratch/vdp/* .
Echo "pushing"
# scp -r ./* atharva2@cc-login1.campuscluster.illinois.edu:/home/atharva2/scratch/vdp/

files=`find . -newermt "-3600 secs"`

for file in $files
do
       sshpass -p "" scp "$file" "atharva2@cc-login1.campuscluster.illinois.edu:/home/atharva2/scratch/vdp/$file"
done
Echo "done!"
